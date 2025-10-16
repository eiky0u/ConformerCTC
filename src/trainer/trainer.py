from pathlib import Path
from typing import List
from contextlib import nullcontext

import pandas as pd
import torch 
import torch.cuda.amp as amp
import torch.nn.functional as F

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer

def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        return torch.cuda.is_bf16_supported()
    # запасной путь: Ampere (SM >= 80) и новее
    major, minor = torch.cuda.get_device_capability()
    return major >= 8


def _pick_amp_dtype(cfg_trainer) -> torch.dtype | None:
    if not cfg_trainer.use_amp or not torch.cuda.is_available():
        return None
    want = cfg_trainer.get("amp_dtype", "auto").lower()
    if want == "bf16":
        return torch.bfloat16
    if want == "fp16":
        return torch.float16
    return torch.bfloat16 if bf16_supported() else torch.float16

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """

        amp_dtype = _pick_amp_dtype(self.cfg_trainer)
        use_amp   = amp_dtype is not None

        scaler = self.scaler
        use_scaler = bool(use_amp and amp_dtype == torch.float16 and scaler is not None)
        
        grad_clip = self.cfg_trainer.grad_clip
        do_clip = bool(grad_clip and grad_clip > 0)

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        ctx = amp.autocast(dtype=amp_dtype) if use_amp else nullcontext()
        if self.is_train:
            with ctx:
                outputs = self.model(**batch)
                batch.update(outputs)
                losses = self.criterion(**batch)            # ожидает logits/logit_length из outputs
                loss = losses["loss"]

            if use_scaler:
                scaler.scale(loss).backward()
                if do_clip:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                if do_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        else:
            with ctx:
                outputs = self.model(**batch)
                batch.update(outputs)
                losses = self.criterion(**batch)      
                loss = losses["loss"]
        
        batch.update(losses)

        metrics.update("loss", float(loss.detach().cpu()))
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
        else:
            # Log Stuff
            self.log_audio(**batch)
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_audio(self, audio, audio_length, **batch):
        sr = 16000
        T = int(audio_length[0])
        self.writer.add_audio(f"audio", audio[0, :, :T], sr)


    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        with torch.no_grad():
            lengths = [int(l) for l in log_probs_length.detach().cpu().tolist()]

            argmax_inds = log_probs.argmax(-1)
            argmax_inds = [seq[:L].detach().cpu().tolist() for seq, L in zip(argmax_inds, lengths)]
            argmax_texts = [self.text_encoder.ctc_decode(seq) for seq in argmax_inds]

            beam_results = self.text_encoder.ctc_beam_search(log_probs, log_probs_length, beam_size=3)
            beam_texts = [results[0][0] if results else "" for results in beam_results]

        rows = {}
        for arg_pred, tgt, beam_pred, ap in list(zip(argmax_texts, text, beam_texts, audio_path))[:examples_to_log]:
            tgt_norm = self.text_encoder.normalize_text(tgt)

            arg_wer = calc_wer(tgt_norm, arg_pred)
            arg_cer = calc_cer(tgt_norm, arg_pred)

            beam_wer = calc_wer(tgt_norm, beam_pred)
            beam_cer = calc_cer(tgt_norm, beam_pred)

            rows[Path(ap).name] = {
                "target": tgt_norm,
                "argmax prediction": arg_pred,
                "beam search prediction": beam_pred,
                "argmax wer": arg_wer,
                "argmax cer": arg_cer,
                "beam search wer": beam_wer,
                "beam search cer": beam_cer,
            }

        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

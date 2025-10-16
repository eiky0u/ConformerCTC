from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any, Optional, List
import json

import pandas as pd
import torch
import torch.cuda.amp as amp
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


def bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        return torch.cuda.is_bf16_supported()
    major, minor = torch.cuda.get_device_capability()
    return major >= 8


def _pick_amp_dtype(cfg) -> Optional[torch.dtype]:
    if not cfg.get("use_amp", False) or not torch.cuda.is_available():
        return None
    want = str(cfg.get("amp_dtype", "auto")).lower()
    if want == "bf16":
        return torch.bfloat16
    if want == "fp16":
        return torch.float16
    return torch.bfloat16 if bf16_supported() else torch.float16


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """
    def __init__(
        self,
        model,
        config,
        device,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        text_encoder,
        save_path: Path | str,
        metrics: Optional[Dict[str, List[Any]]] = None,
        batch_transforms=None,
        skip_model_load: bool = False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms
        self.text_encoder = text_encoder

        self.evaluation_dataloaders = dict(dataloaders)

        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.pred_format = str(self.cfg_trainer.get("pred_format", "jsonl")).lower()
        if self.pred_format not in {"jsonl", "csv"}:
            self.pred_format = "jsonl"
        default_name = f"predictions.{self.pred_format}"
        self.file_name = str(self.cfg_trainer.get("file_name", default_name))

        self.metrics = metrics
        if self.metrics is not None and "inference" in self.metrics:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,  
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            self._from_pretrained(self.cfg_trainer.get("from_pretrained"))


    def run_inference(self) -> Dict[str, Dict[str, float]]:
        """
        Run inference 
        """
        self.is_train = False
        self.model.eval()

        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            if self.evaluation_metrics is not None:
                self.evaluation_metrics.reset()

            out_dir = (self.save_path / part)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / self.file_name

            if self.pred_format == "jsonl" and out_file.exists():
                out_file.unlink()
            rows_accum: List[Dict[str, Any]] = []  
            with torch.no_grad():
                for batch_idx, batch in tqdm(
                    enumerate(dataloader), total=len(dataloader), desc=part
                ):
                    batch = self.process_batch(batch, self.evaluation_metrics)

                    decoded = self._decode_batch(batch)
                    if self.pred_format == "jsonl":
                        with out_file.open("a", encoding="utf-8") as f:
                            for item in decoded:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    else:  # csv
                        rows_accum.extend(decoded)

            if self.pred_format == "csv":
                if rows_accum:
                    pd.DataFrame(rows_accum).to_csv(out_file, index=False, encoding="utf-8")

            logs = self.evaluation_metrics.result() if self.evaluation_metrics is not None else {}
            if logs:
                fmt = " | ".join(f"{k}: {v:.6f}" for k, v in logs.items())
                tqdm.write(f"[{part}] metrics: {fmt}")
            else:
                tqdm.write(f"[{part}] metrics: <none>")
            tqdm.write(f"[{part}] predictions saved to: {out_file}")

            part_logs[part] = logs

        return part_logs


    @torch.no_grad()
    def process_batch(self, batch: Dict[str, Any], metrics: Optional[MetricTracker]) -> Dict[str, Any]:
        amp_dtype = _pick_amp_dtype(self.cfg_trainer)
        ctx = amp.autocast(dtype=amp_dtype) if amp_dtype is not None else nullcontext()

        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        with ctx:
            outputs = self.model(**batch)
            batch.update(outputs)

            if "log_probs" not in batch:
                raise KeyError("Model must return 'log_probs' ([B, T, C]).")
            if "log_probs_length" not in batch:
                if "logit_length" in batch:
                    batch["log_probs_length"] = batch["logit_length"]
                else:
                    raise KeyError("Model must return 'log_probs_length' ([B]).")

            if getattr(self, "criterion", None) is not None:
                losses = self.criterion(**batch)  
                batch.update(losses)
                loss = float(losses["loss"].detach().cpu())
            else:
                loss = None

        if metrics is not None and self.metrics is not None and "inference" in self.metrics:
            if loss is not None:
                metrics.update("loss", loss)
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        return batch

    def _decode_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns list of dicts for writing:
          file, target, greedy, beam, greedy_wer, greedy_cer, beam_wer, beam_cer
        """
        from pathlib import Path as _P

        log_probs = batch["log_probs"]              
        log_probs_length = batch["log_probs_length"]

        text = batch.get("text", [""] * log_probs.shape[0])
        audio_path = batch.get("audio_path", [""] * log_probs.shape[0])

        lengths = [int(l) for l in log_probs_length.detach().cpu().tolist()]

        argmax_inds = log_probs.argmax(-1)  
        argmax_inds = [seq[:L].detach().cpu().tolist() for seq, L in zip(argmax_inds, lengths)]
        greedy_texts = [self.text_encoder.ctc_decode(seq) for seq in argmax_inds]

        beam_size = int(self.cfg_trainer.get("beam_size", 3))
        beam_results = self.text_encoder.ctc_beam_search(log_probs, log_probs_length, beam_size=beam_size)
        beam_texts = [res[0][0] if (res and len(res[0]) > 0) else "" for res in beam_results]

        rows: List[Dict[str, Any]] = []
        for tgt, g, b, ap in zip(text, greedy_texts, beam_texts, audio_path):
            tgt_norm = (
                self.text_encoder.normalize_text(tgt)
                if hasattr(self.text_encoder, "normalize_text")
                else tgt
            )
            rows.append({
                "file": _P(ap).name if ap else "",
                "target": tgt_norm,
                "greedy": g,
                "beam": b,
                "greedy_wer": calc_wer(tgt_norm, g),
                "greedy_cer": calc_cer(tgt_norm, g),
                "beam_wer": calc_wer(tgt_norm, b),
                "beam_cer": calc_cer(tgt_norm, b),
            })
        return rows

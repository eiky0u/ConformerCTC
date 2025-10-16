from typing import Optional, List, Tuple
from collections import defaultdict
import numpy as np
import torch
import re

from .whisper_bpe_encoder import WhisperBPEEncoder

class CTCBPETextEncoder:

    EMPTY_TOK = "^"
    EMPTY_ID = 0

    def __init__(self, pretrained_name: str = "openai/whisper-small", lowercase: bool = False):
        self.bpe = WhisperBPEEncoder(pretrained_name=pretrained_name, lowercase=lowercase)


        self.vocab_size = len(self.bpe)                    
        self.ind2tok = {self.EMPTY_ID: self.EMPTY_TOK}

        for k in range(1, self.vocab_size):
            self.ind2tok[k] = self.bpe.tokenizer.convert_ids_to_tokens(k - 1)

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, item: int) -> str:
        return self.ind2tok[int(item)]


    def encode(self, text: str) -> torch.Tensor:
        return self.bpe.encode(text)

    def decode(self, inds: List[int]) -> str:
        ids = [int(i) for i in inds if int(i) != self.EMPTY_ID]
        return self.bpe.decode_ids(ids)

    def ctc_decode(self, inds: List[int]) -> str:
        prev = None
        kept: List[int] = []
        for i in inds:
            i = int(i)
            if i != self.EMPTY_ID and i != prev:
                kept.append(i)
            prev = i
        return self.bpe.decode_ids(kept)


    @staticmethod
    def _logaddexp(a, b):
        return np.logaddexp(a, b)

    @staticmethod
    def _logsumexp3(a, b, c):
        return np.logaddexp(np.logaddexp(a, b), c)


    def ctc_beam_search_single(
        self,
        log_probs: torch.Tensor,                 
        log_probs_length: Optional[int] = None,
        beam_size: int = 10,
        vocab_top_k: int = 50,                   
    ) -> List[Tuple[str, float]]:


        T, C = log_probs.shape
        if isinstance(log_probs_length, torch.Tensor):
            log_probs_length = int(log_probs_length.item())
        if log_probs_length is None:
            log_probs_length = T

      
        lp = log_probs.detach().float().cpu().numpy()
        blank = self.EMPTY_ID
        NEG_INF = -np.inf

   
        beam = {(): (0.0, NEG_INF)}

        for t in range(log_probs_length):
            y = lp[t]  

            if vocab_top_k is not None and vocab_top_k < C:
                idxs = np.argpartition(y, -vocab_top_k)[-vocab_top_k:]
            else:
                idxs = np.arange(C, dtype=np.int64)

            new_beam = defaultdict(lambda: (NEG_INF, NEG_INF))

            for prefix, (p_b, p_nb) in beam.items():
                nb_pb, nb_pnb = new_beam[prefix]
                nb_pb = self._logsumexp3(nb_pb, p_b + y[blank], p_nb + y[blank])
                new_beam[prefix] = (nb_pb, nb_pnb)

                last = prefix[-1] if prefix else None

                for c in idxs:
                    c = int(c)
                    if c == blank:
                        continue
                    p_c = y[c]

                    if c == last:
                        pb_old, pnb_old = new_beam[prefix]
                        pnb_old = self._logaddexp(pnb_old, p_nb + p_c)
                        new_beam[prefix] = (pb_old, pnb_old)

                        new_prefix = prefix + (c,)
                        pb2, pnb2 = new_beam[new_prefix]
                        pnb2 = self._logaddexp(pnb2, p_b + p_c)
                        new_beam[new_prefix] = (pb2, pnb2)
                    else:
                        new_prefix = prefix + (c,)
                        pb_old, pnb_old = new_beam[new_prefix]
                        pnb_old = self._logaddexp(pnb_old, np.logaddexp(p_b, p_nb) + p_c)
                        new_beam[new_prefix] = (pb_old, pnb_old)

            items = list(new_beam.items())
            items.sort(key=lambda it: np.logaddexp(it[1][0], it[1][1]), reverse=True)
            beam = dict(items[:beam_size])

        hyps: List[Tuple[str, float]] = []
        for prefix, (p_b, p_nb) in beam.items():
            score = float(np.logaddexp(p_b, p_nb))
            text = self.bpe.decode_ids(list(prefix))  
            hyps.append((text, score))

        hyps.sort(key=lambda x: x[1], reverse=True)
        return hyps


    def ctc_beam_search(
        self,
        log_probs_batch: torch.Tensor,                   
        log_probs_lengths: Optional[torch.Tensor] = None,
        beam_size: int = 10,
        vocab_top_k: int = 50,
    ) -> List[List[Tuple[str, float]]]:

        results: List[List[Tuple[str, float]]] = []
        B = log_probs_batch.shape[0]

        lengths = None
        if log_probs_lengths is not None:
            lengths = [int(x) for x in log_probs_lengths.detach().cpu().tolist()]

        for i in range(B):
            T_i = None if lengths is None else lengths[i]
            res_i = self.ctc_beam_search_single(
                log_probs_batch[i],
                T_i,
                beam_size=beam_size,
                vocab_top_k=vocab_top_k,
            )
            results.append(res_i)
        return results


    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
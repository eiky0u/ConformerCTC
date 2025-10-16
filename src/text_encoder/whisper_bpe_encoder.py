from dataclasses import dataclass
from typing import Optional, List
import torch
from transformers import AutoTokenizer

class WhisperBPEEncoder:
    BLANK_ID: int = 0

    def __init__(self, pretrained_name: str = "openai/whisper-small", lowercase: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True)
        self.lowercase = lowercase

    def __len__(self) -> int:
        return int(self.tokenizer.vocab_size) + 1

    def normalize_text(self, s: str) -> str:
        s = s.strip()
        return s.lower() if self.lowercase else s

    def encode(self, text: str) -> torch.Tensor:
        text = self.normalize_text(text)
        hf_ids: List[int] = self.tokenizer.encode(text, add_special_tokens=False)
        rnnt_ids = [i + 1 for i in hf_ids]
        return torch.tensor(rnnt_ids, dtype=torch.long).unsqueeze(0)

    def decode_ids(self, ids: List[int]) -> str:
        hf_ids = [i - 1 for i in ids if i != self.BLANK_ID]
        return self.tokenizer.decode(hf_ids, skip_special_tokens=True).strip()

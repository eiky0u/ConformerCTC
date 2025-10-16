import editdistance
from typing import List, Tuple
from torch import Tensor
from collections import defaultdict


def calc_cer(target_text, predicted_text) -> float:
    return editdistance.eval(target_text, predicted_text) / len(target_text) * 100

def calc_wer(target_text, predicted_text) -> float:
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(target_text.split()) * 100
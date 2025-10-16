import torch
from torch import nn

# --- helpers -----------------------------------------------------------------

def _to_4d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.dim() == 3:        # [B,F,T]
        return x.unsqueeze(1), False
    elif x.dim() == 4:      # [B,*,F,T]
        return x, True
    else:
        raise ValueError(f"Expected [B,F,T] or [B,*,F,T], got {tuple(x.shape)}")

def _per_sample_starts(width: torch.Tensor, dim_size: int) -> torch.Tensor:
    B = width.shape[0]
    max_start = (dim_size - width).clamp(min=1)
    t0 = (torch.rand(B, device=width.device) * max_start.float()).floor().to(torch.long)
    return t0

# --- base --------------------------------------------------------------------

class _MaskBase(nn.Module):
    """
    fill_mode:
      - "const" -> fill_value (например, -80.0 dB) — как паддинг
      - "min"   -> per-sample min по [C,F,T] (удобно с top_db)
      - "mean"  -> per-sample mean по [C,F,T]
    """
    def __init__(self, p: float = 1.0, fill_mode: str = "const", fill_value: float = -80.0):
        super().__init__()
        self.p = p
        self.fill_mode = fill_mode
        self.fill_value = float(fill_value)

    def _get_fill(self, x4d: torch.Tensor) -> torch.Tensor:
        if self.fill_mode == "const":
            return torch.tensor(self.fill_value, dtype=x4d.dtype, device=x4d.device)
        elif self.fill_mode == "min":
            return x4d.amin(dim=(1,2,3), keepdim=True)  # [B,1,1,1]
        elif self.fill_mode == "mean":
            return x4d.mean(dim=(1,2,3), keepdim=True)  # [B,1,1,1]
        else:
            raise ValueError(f"Unknown fill_mode: {self.fill_mode}")

# --- transforms --------------------------------------------------------------

class FreqMask(_MaskBase):
    def __init__(self, max_width: int, num_masks: int = 2, p: float = 1.0,
                 fill_mode: str = "const", fill_value: float = -80.0):
        super().__init__(p=p, fill_mode=fill_mode, fill_value=fill_value)
        self.max_width = max_width
        self.num_masks = num_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.max_width <= 0 or self.num_masks <= 0 or torch.rand(()) > self.p:
            return spec
        x, has_channel = _to_4d(spec)   # [B,C,F,T]
        B, C, F, T = x.shape
        device = x.device

        freq_mask = torch.zeros((B, F), dtype=torch.bool, device=device)
        for _ in range(self.num_masks):
            width = torch.randint(0, min(self.max_width, F) + 1, (B,), device=device)
            if (width == 0).all():
                continue
            f0 = _per_sample_starts(width, F)
            f_idx = torch.arange(F, device=device)[None, :]
            cur = (f_idx >= f0[:, None]) & (f_idx < (f0 + width)[:, None])
            freq_mask |= cur

        if not freq_mask.any():
            return spec

        fill = self._get_fill(x)                       # scalar or [B,1,1,1]
        freq_mask_4d = freq_mask[:, None, :, None]     # [B,1,F,1]
        x = torch.where(freq_mask_4d, fill, x)
        return x if has_channel else x.squeeze(1)


class TimeMask(_MaskBase):
    def __init__(self, max_width: int, num_masks: int = 2, p: float = 1.0,
                 fill_mode: str = "const", fill_value: float = -80.0):
        super().__init__(p=p, fill_mode=fill_mode, fill_value=fill_value)
        self.max_width = max_width
        self.num_masks = num_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.max_width <= 0 or self.num_masks <= 0 or torch.rand(()) > self.p:
            return spec
        x, has_channel = _to_4d(spec)   # [B,C,F,T]
        B, C, F, T = x.shape
        device = x.device

        time_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        for _ in range(self.num_masks):
            width = torch.randint(0, min(self.max_width, T) + 1, (B,), device=device)
            if (width == 0).all():
                continue
            t0 = _per_sample_starts(width, T)
            t_idx = torch.arange(T, device=device)[None, :]
            cur = (t_idx >= t0[:, None]) & (t_idx < (t0 + width)[:, None])
            time_mask |= cur

        if not time_mask.any():
            return spec

        fill = self._get_fill(x)
        time_mask_4d = time_mask[:, None, None, :]     # [B,1,1,T]
        x = torch.where(time_mask_4d, fill, x)
        return x if has_channel else x.squeeze(1)


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_max_width: int = 10,
        time_max_width: int = 40,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 1.0,
        fill_mode: str = "const", 
        fill_value: float = -80.0,     
        p_freq: float = 0.9,
        p_time: float = 0.9,
    ):
        super().__init__()
        self.p = p
        self.freq = FreqMask(freq_max_width, n_freq_masks, p=p_freq,
                             fill_mode=fill_mode, fill_value=fill_value)
        self.time = TimeMask(time_max_width, n_time_masks, p=p_time,
                             fill_mode=fill_mode, fill_value=fill_value)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(()) > self.p:
            return spec
        x = self.freq(spec)
        x = self.time(x)
        return x

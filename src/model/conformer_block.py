import torch
from torch import nn
import torch.nn.functional as F
import math


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, p: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            groups=d_model, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.do = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x.transpose(1, 2)
        x = self.pw1(x)
        x = self.glu(x)
        x = self.dw(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pw2(x)
        x = self.do(x)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20000, p: float = 0.0) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)          
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe, persistent=False)
        self.do = nn.Dropout(p)

    def forward(self, length: int) -> torch.Tensor:
        return self.do(self.pe[:, :length, :])


class AttentionModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, p: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=p,
            batch_first=True,
            bias=True
        )
        self.pe = PositionalEncoding(d_model, p=0.0)
        self.do_out = nn.Dropout(p)

    def forward(
        self,
        x: torch.Tensor,                              
        key_padding_mask: torch.Tensor | None = None, 
        attn_mask: torch.Tensor | None = None        
    ) -> torch.Tensor:
        
        B, T, D = x.shape
        x = self.ln(x)
        x = x + self.pe(T)  

        out, _ = self.mha(
            query=x, key=x, value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask                 
        )
        return self.do_out(out)



class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, kernel_size: int,
                 n_heads: int, p_ff: float, p_conv: float, p_mhsa: float) -> None:
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, p_ff)
        self.att = AttentionModule(d_model, n_heads, p_mhsa)
        self.conv = ConvolutionModule(d_model, kernel_size, p_conv)
        self.ff2 = FeedForward(d_model, d_ff, p_ff)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,   
        attn_mask: torch.Tensor | None = None           
    ) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.att(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.ln_out(x)
        return x

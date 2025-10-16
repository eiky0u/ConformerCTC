import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import math

from src.model.conformer_block import ConformerBlock

class ConvSubsampling2D(nn.Module):
    def __init__(self, n_mels: int, d_model: int, p_sub: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model // 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        Fp = int(self.transform_lengths(n_mels))
        self.out_proj = nn.Linear(d_model * Fp, d_model)
        self.dropout = nn.Dropout(p_sub)

    @staticmethod
    def transform_lengths(t: torch.Tensor | int) -> torch.Tensor:
        return ((t + 1) // 2 + 1) // 2
    

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = spec.unsqueeze(1)                           
        x = self.conv(x)                                
        B, C, Fp, Tp = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()          
        x = x.view(B, Tp, C * Fp)                       
        x = self.out_proj(x)                            
        x = self.dropout(x)
        return x


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,     
        num_blocks: int = 16,
        d_model: int = 256,
        d_ff: int = 1024,
        kernel_size: int = 7,
        n_heads: int = 4,
        p_ff: float = 0.1,
        p_conv: float = 0.1,
        p_mhsa: float = 0.1,
        p_sub: float = 0.1,
    ):
        super().__init__()
        self.conv_kernel_size = kernel_size

        self.subsampling = ConvSubsampling2D(n_mels=n_mels, d_model=d_model, p_sub=p_sub)

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                d_ff=d_ff,
                kernel_size=kernel_size,
                n_heads=n_heads,
                p_ff=p_ff,
                p_conv=p_conv,
                p_mhsa=p_mhsa,
            )
            for _ in range(num_blocks)

        ])
    
    def transform_input_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return self.subsampling.transform_lengths(input_lengths)
    

    def forward(self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch):
        x = self.subsampling(spectrogram)
        out_lens = self.transform_input_lengths(spectrogram_length)

        B, Tprime, _ = x.shape
        idx = torch.arange(Tprime, device=x.device).unsqueeze(0).expand(B, -1)
        key_padding_mask = idx >= out_lens.unsqueeze(1)   

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask, attn_mask=None)

        return x, out_lens
    


class ConformerCTC(nn.Module):
    def __init__(self,  n_mels=64, n_tokens=28, d_model=256, d_ff=1024,
                num_blocks=12, n_heads=4, conv_kernel_size=7,
                p_ff=0.1, p_conv=0.1, p_mhsa=0.1, p_sub=0.1, blank_id=0):
        
        super().__init__()
        self.encoder = ConformerEncoder(
            n_mels=n_mels, num_blocks=num_blocks, d_model=d_model, d_ff=d_ff,
            kernel_size=conv_kernel_size, n_heads=n_heads,
            p_ff=p_ff, p_conv=p_conv, p_mhsa=p_mhsa, p_sub=p_sub
        )
        self.proj = nn.Linear(d_model, n_tokens)
        self.blank_id = blank_id


    def forward(self, spectrogram, spectrogram_length,
                text_encoded, text_encoded_length, **batch) -> torch.Tensor:

        x, lenghts = self.encoder(spectrogram, spectrogram_length,)

        logits = self.proj(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": lenghts}
    
    
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
    

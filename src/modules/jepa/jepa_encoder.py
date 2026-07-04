import torch
import torch.nn as nn
from src.modules.core.encoder import TransformerBlock

class JEPAEncoder(nn.Module):

    def __init__(self, d_final: int, stage:int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.2):
        super().__init__()
        self.transformers = nn.Sequential(*[TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(stage)])
        self.proj = nn.Linear(d_model, d_final)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        key_padding_mask = (attention_mask == 0)

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.proj(x)

        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        return x 
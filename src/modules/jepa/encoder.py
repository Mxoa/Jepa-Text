# src/modules/jepa/models/encoders/transformer_encoder.py

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Un seul bloc transformer : pre-norm, MHSA, résiduel, pre-norm, MLP, résiduel.

    Pre-norm (norm AVANT attention) plutôt que post-norm :
    plus stable à entraîner, c'est ce que font les modèles modernes (LLaMA etc.)

    Architecture par bloc :
        x → LayerNorm → MHSA → + x  (résiduel)
          → LayerNorm → MLP  → + x  (résiduel)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # --- Self-attention -------------------------------------------------
        # On délègue à PyTorch : nn.MultiheadAttention gère Q/K/V projections,
        # scaled dot-product, et output projection.
        # batch_first=True : les tenseurs sont (B, L, D) et non (L, B, D)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # --- MLP (Feed-Forward) ---------------------------------------------
        # Deux linéaires avec GELU au milieu.
        # d_ff est typiquement 4 * d_model (convention transformer original).
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        """
        Args:
            x                : (B, L, D)
            key_padding_mask : (B, L) booléen, True aux positions à IGNORER
                               (convention nn.MultiheadAttention : True = ignore)
                               On convertit notre attention_mask (1=valide, 0=pad)
                               en amont avant d'appeler ce bloc.
        Returns:
            (B, L, D)
        """
        # Pre-norm + self-attention + résiduel
        # On passe la même séquence en query, key, value (self-attention)
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,   # on n'a pas besoin des poids d'attention pour l'instant
        )
        x = x + self.dropout(attn_out)

        # Pre-norm + MLP + résiduel
        x = x + self.ff(self.norm2(x))

        return x


class TransformerEncoder(nn.Module):
    """
    Encodeur complet : embedding + N blocs transformer.

    Prend des input_ids et un attention_mask (convention HuggingFace :
    1 = token valide, 0 = padding) et produit des représentations contextuelles.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id

        # --- Embeddings -----------------------------------------------------
        # token_embedding     : mappe chaque token id → vecteur D
        # position_embedding  : mappe chaque position 0..L-1 → vecteur D
        # Les deux sont APPRIS (pas sinusoïdal, pas RoPE — on garde simple)
        self.token_embedding    = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.embed_dropout = nn.Dropout(dropout)

        # --- Blocs transformer ----------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Norm finale (bonne pratique : stabilise les représentations en sortie)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        Args:
            input_ids      : (B, L)  ids de tokens
            attention_mask : (B, L)  1 = token valide, 0 = padding
                             (convention HuggingFace, celle de datamodule)

        Returns:
            hidden_states  : (B, L, D)  représentations contextuelles
        """
        B, L = input_ids.shape

        # --- Embedding ------------------------------------------------------
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)  # (1, L)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # --- Conversion du masque -------------------------------------------
        # nn.MultiheadAttention attend key_padding_mask booléen :
        #     True  → IGNORER cette position (padding)
        #     False → garder cette position
        # Notre attention_mask HuggingFace est l'inverse :
        #     1 → valide, 0 → padding
        # Donc on inverse : key_padding_mask = (attention_mask == 0)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # (B, L), bool

        # --- Blocs transformer ----------------------------------------------
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        return self.final_norm(x)  # (B, L, D)
    
if __name__ == '__main__':
    encoder = TransformerEncoder(10, 2, 2, 2, 8, 8)

    batch = torch.tensor([
        [1, 2, 3, 4, 0, 0, 0],
        [1, 1, 3, 6, 6, 0, 0]
    ]) # batch de taille 2, on a deux séquence de taille 7

    attn_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0]
    ])

    output = encoder.forward(batch, attn_mask)

    print(output)
    print(output.shape)
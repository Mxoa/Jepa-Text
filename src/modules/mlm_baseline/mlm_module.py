# src/modules/mlm_baseline/mlm_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class MLMHead(nn.Module):
    """
    Tête de prédiction MLM : projette les représentations D → vocab_size.
    
    On l'applique SEULEMENT aux positions masquées (pas à toute la séquence),
    ce qui force à gérer le gather — exactement la même logique qu'on
    réutilisera dans JEPA pour gather les représentations cibles.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states : (N, D)  représentations déjà gatherées aux positions masquées
        Returns:
            logits        : (N, vocab_size)
        """
        return self.proj(self.norm(hidden_states))


class MLMModule(L.LightningModule):
    """
    LightningModule pour le pré-entraînement MLM.

    Reçoit les batches de TextDataModule :
        input_ids        : (B, L)
        attention_mask   : (B, L)
        masked_positions : (B, M)   — -1 = padding de position
        target_ids       : (B, M)   — -1 = padding de cible

    Étapes dans training_step :
        1. Encoder  → hidden_states (B, L, D)
        2. Gather   → représentations aux positions masquées valides (N, D)
        3. MLMHead  → logits (N, vocab_size)
        4. Loss     → cross-entropy sur les N positions valides seulement
    """

    def __init__(
        self,
        encoder: nn.Module,          # ton TransformerEncoder
        vocab_size: int,
        d_model: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])  # sauvegarde lr, etc. dans le checkpoint

        self.encoder = encoder
        self.head    = MLMHead(d_model, vocab_size)
        self.lr = lr
        self.weight_decay = weight_decay

    def _gather_masked_positions(
        self,
        hidden_states: torch.Tensor,   # (B, L, D)
        masked_positions: torch.Tensor, # (B, M)  — -1 = padding
        attention_mask: torch.Tensor,   # (B, L)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather les représentations aux positions masquées valides,
        en gérant le left-padding et le sentinel -1.

        Retourne :
            gathered : (N, D)   représentations des positions masquées valides
            valid    : (B, M)   masque booléen des positions réelles (non -1)
        """
        B, L, D = hidden_states.shape
        M = masked_positions.shape[1]

        valid = masked_positions != -1   # (B, M)

        pad_len = (attention_mask == 0).sum(dim=1, keepdim=True)  # (B, 1)
        
        corrected = (masked_positions + pad_len).clamp(min=0)     # (B, M)

        idx = corrected.unsqueeze(-1).expand(-1, -1, D)           # (B, M, D)
        gathered_all = hidden_states.gather(dim=1, index=idx)     # (B, M, D)

        gathered = gathered_all[valid]                             # (N, D)

        return gathered, valid

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        hidden_states = self.encoder(
            batch["input_ids"],
            batch["attention_mask"],
        )  # (B, L, D)

        gathered, valid = self._gather_masked_positions(
            hidden_states,
            batch["masked_positions"],
            batch["attention_mask"],
        )  # gathered : (N, D)

        logits = self.head(gathered)              # (N, vocab_size)

        targets = batch["target_ids"][valid] #(N,)

        loss = F.cross_entropy(logits, targets)

        self.log(f"{stage}/loss", loss,
                 on_step=(stage == "train"),
                 on_epoch=True,
                 prog_bar=True)
        self.log(f"{stage}/perplexity", torch.exp(loss),
                 on_step=True,
                 on_epoch=True)

        return loss


    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        decay_params     = [p for n, p in self.named_parameters()
                            if p.requires_grad and not _no_decay(n)]
        no_decay_params  = [p for n, p in self.named_parameters()
                            if p.requires_grad and _no_decay(n)]

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.lr,
        )
        return optimizer


def _no_decay(param_name: str) -> bool:
    """Biais et paramètres de normalisation exclus du weight decay."""
    return any(nd in param_name for nd in ["bias", "norm.weight"])
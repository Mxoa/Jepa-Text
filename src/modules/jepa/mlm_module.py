# src/modules/jepa/models/mlm_module.py

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
        # Projection linéaire + LayerNorm avant (convention BERT)
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

    # ------------------------------------------------------------------
    # Logique commune train / val
    # ------------------------------------------------------------------

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

        # Masque des positions réelles (pas le padding -1)
        valid = masked_positions != -1   # (B, M)

        # Correction left-padding : masked_positions est dans l'espace
        # de la séquence non-paddée, mais notre tenseur hidden_states
        # est de longueur L avec du padding à gauche.
        # pad_len[i] = nombre de tokens de padding à gauche pour l'exemple i
        pad_len = (attention_mask == 0).sum(dim=1, keepdim=True)  # (B, 1)
        
        # Positions corrigées dans le tenseur paddé
        # On clamp à 0 pour les -1 (ils seront ignorés via `valid`)
        corrected = (masked_positions + pad_len).clamp(min=0)     # (B, M)

        # Expand pour gather sur la dim D
        idx = corrected.unsqueeze(-1).expand(-1, -1, D)           # (B, M, D)
        gathered_all = hidden_states.gather(dim=1, index=idx)     # (B, M, D)

        # Ne garder que les positions valides → (N, D)
        gathered = gathered_all[valid]                             # (N, D)

        return gathered, valid

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        # 1. Encodeur
        hidden_states = self.encoder(
            batch["input_ids"],
            batch["attention_mask"],
        )  # (B, L, D)

        # 2. Gather aux positions masquées
        gathered, valid = self._gather_masked_positions(
            hidden_states,
            batch["masked_positions"],
            batch["attention_mask"],
        )  # gathered : (N, D)

        # 3. Tête MLM → logits
        logits = self.head(gathered)              # (N, vocab_size)

        # 4. Cibles : target_ids aux positions valides seulement
        targets = batch["target_ids"][valid]      # (N,)

        # 5. Cross-entropy
        loss = F.cross_entropy(logits, targets)

        # 6. Logging  —  on_step=True affiche à chaque step dans la barre
        #                on_epoch=True moyenne sur l'epoch entière
        self.log(f"{stage}/loss", loss,
                 on_step=(stage == "train"),
                 on_epoch=True,
                 prog_bar=True)

        # Perplexité : métrique naturelle pour du MLM, exp(loss CE)
        # Plus lisible que la loss brute pour savoir "est-ce que ça apprend"
        self.log(f"{stage}/perplexity", torch.exp(loss),
                 on_step=True,
                 on_epoch=True)

        return loss

    # ------------------------------------------------------------------
    # Hooks Lightning
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # AdamW : Adam + weight decay découplé (standard pour les transformers)
        # On exclut les biais et LayerNorm du weight decay (convention BERT)
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
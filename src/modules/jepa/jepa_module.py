# src/modules/jepa/models/jepa_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoModel

from src.modules.core.encoder import TransformerBlock


# ══════════════════════════════════════════════════════════════════════════════
#  1. Table d'embeddings BERT gelée
# ══════════════════════════════════════════════════════════════════════════════

class FrozenBertEmbedding(nn.Module):
    """
    Extrait uniquement la table word_embeddings de BERT pré-entraîné.
    token_ids (B, L) → vecteurs statiques (B, L, 768).
    Aucun gradient — jamais mis à jour.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        bert = AutoModel.from_pretrained(model_name)
        self.embedding = nn.Embedding.from_pretrained(
            bert.embeddings.word_embeddings.weight.detach().clone(),
            freeze=True,
            padding_idx=0,
        )
        self.hidden_size = self.embedding.embedding_dim  # 768

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (B, L) → (B, L, 768)
        return self.embedding(token_ids)


# ══════════════════════════════════════════════════════════════════════════════
#  2. JEPAEncoder entraînable
# ══════════════════════════════════════════════════════════════════════════════

class JEPAEncoder(nn.Module):
    """
    Prend des token embeddings contextualisés et produit UN vecteur de phrase.

    (B, L, d_model) → N blocs transformer → mean pool → projection → (B, d_final)

    C'est l'unique module entraînable du pipeline.
    """

    def __init__(
        self,
        d_model: int,       # doit correspondre à hidden_size du backbone (768)
        d_final: int,       # dimension de l'espace latent cible
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_final)

    def forward(
        self,
        x: torch.Tensor,               # (B, L, d_model)
        attention_mask: torch.Tensor,  # (B, L) — 1=valide, 0=padding
    ) -> torch.Tensor:                 # (B, d_final)

        key_padding_mask = (attention_mask == 0)  # convention nn.MHA

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)

        # Mean pooling sur les tokens valides seulement
        mask = attention_mask.unsqueeze(-1).float()       # (B, L, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return self.proj(x)   # (B, d_final)


# ══════════════════════════════════════════════════════════════════════════════
#  3. Prédicteur
# ══════════════════════════════════════════════════════════════════════════════

class JEPAPredictor(nn.Module):
    """
    Prédit le vecteur cible depuis le vecteur contexte.
    MLP simple : contexte (B, d_final) → cible prédite (B, d_final).
    Intentionnellement plus léger que l'encodeur.
    """

    def __init__(self, d_final: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_final, d_final * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_final * 2, d_final),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, d_final) → (B, d_final)
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  4. LightningModule JEPA
# ══════════════════════════════════════════════════════════════════════════════

class JEPAModule(L.LightningModule):
    """
    Pipeline JEPA complet.

    Pour chaque batch issu du TextDataModule :

        input_ids        : (B, L)  — séquence avec [MASK] aux positions masquées
        attention_mask   : (B, L)
        masked_positions : (B, M)  — positions masquées (-1 = padding)
        target_ids       : (B, M)  — tokens originaux aux positions masquées

    Étapes du forward :

        1. Reconstruit la séquence originale (remplace [MASK] par target_ids)
        2. Embedding BERT gelé sur contexte ET cible
        3. JEPAEncoder → vecteur contexte + vecteur cible
        4. Prédicteur → vecteur cible prédit
        5. Loss = cosine distance(prédit, cible)
    """

    def __init__(
        self,
        backbone_name: str = "bert-base-uncased",
        d_final: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Embedding gelé — partagé contexte et cible
        self.embedder  = FrozenBertEmbedding(backbone_name)
        d_model        = self.embedder.hidden_size  # 768

        # Encodeur partagé contexte / cible (mêmes poids)
        self.encoder   = JEPAEncoder(d_model, d_final, n_layers, n_heads, d_ff, dropout)

        # Prédicteur : contexte → cible prédite
        self.predictor = JEPAPredictor(d_final, dropout)

    # ── Reconstruction de la séquence originale ──────────────────────────────

    def _reconstruct_target_ids(
        self,
        input_ids: torch.Tensor,        # (B, L)  — avec [MASK]
        masked_positions: torch.Tensor, # (B, M)  — -1 = padding
        target_ids: torch.Tensor,       # (B, M)
    ) -> torch.Tensor:                  # (B, L)  — original sans [MASK]
        """
        Remet les tokens originaux à leurs positions pour encoder la cible.
        Travaille sur une copie — n'écrase pas input_ids.
        """
        original = input_ids.clone()
        valid    = masked_positions != -1  # (B, M)

        for b in range(input_ids.size(0)):
            pos = masked_positions[b][valid[b]]   # positions réelles
            tgt = target_ids[b][valid[b]]         # tokens originaux
            original[b, pos] = tgt

        return original

    # ── Étape commune train / val ─────────────────────────────────────────────

    def _step(self, batch: dict, stage: str) -> torch.Tensor:

        input_ids        = batch["input_ids"]         # (B, L) avec [MASK]
        attention_mask   = batch["attention_mask"]    # (B, L)
        masked_positions = batch["masked_positions"]  # (B, M)
        target_ids_batch = batch["target_ids"]        # (B, M)

        # 1. Séquence originale pour la cible
        original_ids = self._reconstruct_target_ids(
            input_ids, masked_positions, target_ids_batch
        )

        # 2. Embeddings BERT gelés (pas de gradient ici)
        ctx_emb = self.embedder(input_ids)    # (B, L, 768)
        tgt_emb = self.embedder(original_ids) # (B, L, 768)

        # 3. Encodeur partagé → vecteurs de phrase
        ctx_vec = self.encoder(ctx_emb, attention_mask)  # (B, d_final)
        tgt_vec = self.encoder(tgt_emb, attention_mask)  # (B, d_final)

        # Détache la cible : on n'optimise pas l'encodeur "côté cible"
        # (évite que les deux représentations ne s'effondrent
        #  vers le même vecteur par le chemin le plus court)
        tgt_vec = tgt_vec.detach()

        # 4. Prédicteur
        pred_vec = self.predictor(ctx_vec)  # (B, d_final)

        # 5. Loss cosinus (entre -1 et 1, on veut minimiser la distance)
        loss = 1 - F.cosine_similarity(pred_vec, tgt_vec, dim=-1)
        loss = loss.mean()

        # Métriques de collapse : si la variance est proche de 0
        # tous les vecteurs convergent vers le même point
        with torch.no_grad():
            variance = tgt_vec.var(dim=0).mean()

        self.log(f"{stage}/loss",     loss,     on_step=(stage=="train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}/variance", variance, on_step=False,            on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # Seulement les params entraînables (encodeur + prédicteur)
        # L'embedder gelé n'a pas de requires_grad=True donc AdamW l'ignore
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    # ── Extraction d'embedding au moment de l'éval ───────────────────────────

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Utilisé à l'éval (MTEB, nearest neighbours, STS-B).
        Retourne le vecteur de phrase normalisé L2.
        """
        self.eval()
        emb = self.embedder(input_ids)
        vec = self.encoder(emb, attention_mask)
        return F.normalize(vec, dim=-1)   # (B, d_final)
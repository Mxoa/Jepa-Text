from modules.core.encoder import TransformerEncoder
from modules.mlm_baseline.mlm_module import MLMModule
from data.tokenizer import TokenizerWrapper
import torch
import numpy as np
from torch.utils.data import DataLoader
# eval/sts_eval.py
import torch
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from transformers import AutoTokenizer


# ── 1. Charge le checkpoint ──────────────────────────────────────────────────

CKPT   = "./logs/mlm_baseline/colab/epoch=13-step=25648.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

encoder = TransformerEncoder(
    vocab_size=30522, d_model=128, n_heads=4,
    n_layers=3, d_ff=512, max_seq_len=128,
)
module = MLMModule.load_from_checkpoint(
    CKPT, encoder=encoder, vocab_size=30522, d_model=128
)
module.eval().to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ── 2. Fonction d'embedding ──────────────────────────────────────────────────

def embed(sentences: list[str], batch_size: int = 64) -> np.ndarray:
    all_embs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
        ids   = enc["input_ids"].to(DEVICE)
        mask  = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            hidden = module.encoder(ids, mask)          # (B, L, D)

        # mean pooling
        expanded = mask.unsqueeze(-1).float()
        emb = (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)
        all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs)

# ── 3. Charge STS-B ─────────────────────────────────────────────────────────

ds = load_dataset("mteb/stsbenchmark-sts", split="test")
# colonnes : sentence1, sentence2, score (0-5)

sentences1 = ds["sentence1"]
sentences2 = ds["sentence2"]
gold       = np.array(ds["score"], dtype=float)

# ── 4. Calcule les embeddings et la corrélation ──────────────────────────────

emb1 = embed(sentences1)
emb2 = embed(sentences2)

# similarité cosinus pour chaque paire
norm1     = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
norm2     = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
cos_sims  = (norm1 * norm2).sum(axis=1)

spearman, pvalue = spearmanr(cos_sims, gold)

print(f"\nSTS-Benchmark — test split")
print(f"  Spearman ρ : {spearman:.4f}")
print(f"  p-value    : {pvalue:.2e}")
print(f"  N paires   : {len(gold)}")

# Points de repère pour interpréter
print(f"\n  Référence :")
print(f"    BERT mean-pool sans fine-tuning : ~0.47")
print(f"    GloVe moyenne                   : ~0.58")
print(f"    Sentence-BERT fine-tuné         : ~0.84")
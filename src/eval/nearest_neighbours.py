PROBE_SENTENCES = [
    # Amitié / liens humains
    "She laughed with her best friend until they cried.",
    "They had known each other since childhood and trusted each other completely.",
    "He called his old friend just to hear a familiar voice.",
    "True friendship means being there even during the hard times.",
    "They sat together in comfortable silence, no words needed.",

    # Jeu / enfance
    "The children ran across the playground chasing each other.",
    "He spent the afternoon building a tower of blocks.",
    "She rolled the dice and moved her piece across the board.",
    "The kids invented a game with sticks and fallen leaves.",
    "They played until the sun went down and their mothers called them home.",

    # Nature / paysage
    "The river ran quietly through the valley at dawn.",
    "Tall pine trees lined the mountain path on both sides.",
    "A light fog covered the meadow in the early morning.",
    "The leaves turned red and orange as autumn arrived.",
    "She watched the waves break gently against the rocky shore.",

    # Tristesse / perte
    "He sat alone in the empty room after she left.",
    "The news arrived like a cold wind no one expected.",
    "She stared at the old photograph for a long time.",
    "Something felt permanently missing after that day.",
    "He could not find the words to explain what he had lost.",

    # Technologie / abstrait
    "The algorithm processed millions of data points per second.",
    "Neural networks learn patterns through repeated exposure to examples.",
    "The server returned an unexpected error during inference.",
    "She debugged the pipeline until the embeddings looked correct.",
    "Training a transformer requires careful tuning of the learning rate.",

    # Nourriture (cluster neutre de contrôle)
    "She baked a warm loaf of bread on Sunday morning.",
    "The soup simmered slowly on the stove all afternoon.",
    "He cut a slice of cake and handed it to her.",
    "The market was full of fresh vegetables and ripe fruit.",
    "They shared a meal and talked about everything and nothing.",
]

# eval/nearest_neighbours.py
import torch
import numpy as np
from transformers import AutoTokenizer
from modules.core.encoder import TransformerEncoder
from modules.mlm_baseline.mlm_module import MLMModule

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

def embed(sentences):
    enc  = tokenizer(sentences, padding=True, truncation=True,
                     max_length=128, return_tensors="pt")
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        hidden = module.encoder(ids, mask)
    expanded = mask.unsqueeze(-1).float()
    emb = (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)
    emb = emb.cpu().numpy()
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)  # L2-normalize

embs = embed(PROBE_SENTENCES)

# Pour chaque phrase, affiche les 3 plus proches voisins
print("\n" + "=" * 70)
for i, query in enumerate(PROBE_SENTENCES):
    sims   = embs @ embs[i]           # cosinus (déjà normalisé)
    sims[i] = -1                      # exclut la phrase elle-même
    top3   = np.argsort(sims)[::-1][:3]

    print(f"\nQuery : {query}")
    for rank, j in enumerate(top3):
        print(f"  #{rank+1} ({sims[j]:.3f}) {PROBE_SENTENCES[j]}")
print("=" * 70)
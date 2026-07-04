from transformers import AutoModel
import torch

# Charge BERT juste pour extraire la table d'embeddings
bert = AutoModel.from_pretrained("bert-base-uncased")

# La table d'embeddings : (30522, 768)
embedding_table = bert.embeddings.word_embeddings.weight.detach().clone()

# Crée un nn.Embedding initialisé avec ces poids
embedding = torch.nn.Embedding.from_pretrained(
    embedding_table,
    freeze=True,         # gelé — pas de gradient
    padding_idx=0,
)

print(embedding.weight.shape)         # (30522, 768)
print(embedding.weight.requires_grad) # False

# Test
ids = torch.tensor([[101, 1996, 3007, 102]])  # [CLS] the capital [SEP]
out = embedding(ids)
print(out.shape)                      # (1, 4, 768)
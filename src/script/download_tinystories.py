
# scripts/download_tinystories.py

from datasets import load_dataset
import os

os.makedirs("data/raw", exist_ok=True)

print("Téléchargement TinyStories...")
ds = load_dataset("roneneldan/TinyStories")

print("Sauvegarde train.parquet...")
ds["train"].to_parquet("data/raw/tinystories_train.parquet")

print("Sauvegarde val.parquet...")
ds["validation"].to_parquet("data/raw/tinystories_val.parquet")

print(f"Train : {len(ds['train']):,} exemples")
print(f"Val   : {len(ds['validation']):,} exemples")
print("Fini → data/raw/")
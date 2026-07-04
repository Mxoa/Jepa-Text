# src/modules/jepa/train_jepa.py

import os
import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from data.datamodule import TextDataModule
from src.modules.jepa.jepa_module import JEPAModule


def build_datamodule(args) -> TextDataModule:
    return TextDataModule(
        train_path=args.train_path,
        val_path=args.val_path,
        tokenizer_name=args.backbone,
        window_size=args.window_size,
        stride=args.stride,
        alpha=args.alpha,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def build_module(args) -> JEPAModule:
    return JEPAModule(
        backbone_name=args.backbone,
        d_final=args.d_final,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def build_callbacks(args) -> list:
    return [
        # Sauvegarde le meilleur checkpoint selon val/loss
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="jepa-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        ),
        # Arrête si val/loss ne baisse plus depuis N epochs
        EarlyStopping(
            monitor="val/loss",
            patience=args.patience,
            mode="min",
            verbose=True,
        ),
        # Log le learning rate à chaque step (utile avec cosine scheduler)
        LearningRateMonitor(logging_interval="step"),
    ]


def build_logger(args):
    if args.logger == "wandb":
        return WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            save_dir=args.output_dir,
        )
    return TensorBoardLogger(
        save_dir=args.output_dir,
        name=args.run_name,
    )


def train(args):
    L.seed_everything(args.seed)

    dm     = build_datamodule(args)
    module = build_module(args)

    # Résumé des params entraînables avant de lancer
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"\nParams total      : {total:>12,}")
    print(f"Params entraînable: {trainable:>12,}  ({100*trainable/total:.1f}%)")
    print(f"Backbone gelé     : {total-trainable:>12,}\n")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,           # "16-mixed" sur GPU, "32" sur CPU
        logger=build_logger(args),
        callbacks=build_callbacks(args),
        gradient_clip_val=1.0,              # indispensable pour les transformers
        log_every_n_steps=args.log_every,
        val_check_interval=args.val_interval,
    )

    # Reprend depuis un checkpoint si fourni
    trainer.fit(module, datamodule=dm, ckpt_path=args.resume)

    print(f"\nMeilleur checkpoint : {trainer.checkpoint_callback.best_model_path}")
    print(f"Meilleure val/loss  : {trainer.checkpoint_callback.best_model_score:.4f}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Entraîne le modèle JEPA")

    # Données
    g = p.add_argument_group("Données")
    g.add_argument("--train_path",   required=True)
    g.add_argument("--val_path",     required=True)
    g.add_argument("--window_size",  type=int,   default=128)
    g.add_argument("--stride",       type=int,   default=None)
    g.add_argument("--alpha",        type=float, default=0.30)
    g.add_argument("--batch_size",   type=int,   default=64)
    g.add_argument("--num_workers",  type=int,   default=2)

    # Modèle
    g = p.add_argument_group("Modèle")
    g.add_argument("--backbone",     default="bert-base-uncased")
    g.add_argument("--d_final",      type=int,   default=256)
    g.add_argument("--n_layers",     type=int,   default=3)
    g.add_argument("--n_heads",      type=int,   default=4)
    g.add_argument("--d_ff",         type=int,   default=512)
    g.add_argument("--dropout",      type=float, default=0.1)

    # Optimisation
    g = p.add_argument_group("Optimisation")
    g.add_argument("--lr",           type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=1e-2)
    g.add_argument("--max_epochs",   type=int,   default=20)
    g.add_argument("--patience",     type=int,   default=5)
    g.add_argument("--precision",    default="16-mixed")

    # Infra
    g = p.add_argument_group("Infra")
    g.add_argument("--output_dir",     default="logs/jepa")
    g.add_argument("--run_name",       default="jepa_baseline")
    g.add_argument("--logger",         choices=["wandb", "tensorboard"], default="tensorboard")
    g.add_argument("--wandb_project",  default="text-jepa")
    g.add_argument("--log_every",      type=int,   default=50)
    g.add_argument("--val_interval",   type=float, default=0.5)  # val 2x par epoch
    g.add_argument("--seed",           type=int,   default=42)
    g.add_argument("--resume",         default=None, help="Chemin vers un checkpoint pour reprendre")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
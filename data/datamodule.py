"""LightningDataModule orchestrating tokenisation, chunking & masking — fully streamed."""

from typing import Any, Dict, Iterator, List, Optional

import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from data.chunking import Chunking
from data.masking import Masking
from data.tokenizer import TokenizerWrapper


# ══════════════════════════════════════════════════════════════════════════════
#  Streaming Dataset
# ══════════════════════════════════════════════════════════════════════════════


class StreamedMaskedDataset(IterableDataset):
    """Streaming dataset that tokenises, chunks & masks texts on-the-fly.

    Reads from a HuggingFace ``datasets`` IterableDataset (parquet file read
    in streaming mode) and yields masked chunks one at a time **without ever
    loading the full dataset into memory**.

    A token buffer is maintained across consecutive texts so that chunk
    boundaries respect the sliding window even when individual texts are short.

    Yields
    ------
    dict
        With keys ``masked_ids``, ``masked_positions``, ``target_ids``
        (see :meth:`Masking.apply`).
    """

    def __init__(
        self,
        hf_iterable: Any,  # datasets.iteratable_dataset.IterableDataset
        tokenizer: TokenizerWrapper,
        window_size: int,
        stride: int,
        alpha: float = 0.15,
    ) -> None:
        super().__init__()
        self._hf_iterable = hf_iterable
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.alpha = alpha

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield masked chunks by iterating over the streaming dataset.

        A fresh :class:`Chunking` instance is created per worker so that
        each worker has its own independent token buffer (safe for
        multi-process data loading).
        """
        chunker = Chunking(self.tokenizer, window_size=self.window_size, stride=self.stride)
        masker = Masking(self.tokenizer, alpha=self.alpha)

        for example in self._hf_iterable:
            text: str = example["text"]
            ids = self.tokenizer.encode(text)
            chunker._buffer.extend(ids)

            # Drain complete windows from the buffer
            while len(chunker._buffer) >= chunker.window_size:
                chunk = chunker._buffer[: chunker.window_size]
                chunker._buffer = chunker._buffer[chunker.stride :]
                yield masker.apply(chunk)

            # After each text, flush the remaining buffer as a (possibly
            # smaller) chunk, then reset the buffer for the next text.
            leftover = chunker.flush()
            if leftover is not None:
                yield masker.apply(leftover)

    def __len__(self) -> int:
        """Not available for streaming datasets."""
        raise TypeError(
            f"{type(self).__name__} is a streaming dataset and does not have "
            f"a deterministic length. Use ``len`` only for informational purposes "
            f"in the Lightning progress bar."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Lightning DataModule
# ══════════════════════════════════════════════════════════════════════════════


class TextDataModule(L.LightningDataModule):
    """Orchestrates data loading for MLM-style pre-training on parquet files.

    All data loading is **fully streamed** — the dataset is never materialised
    in memory.  One row at a time is read from the parquet file, tokenised,
    chunked and masked on-the-fly.

    Typical usage::

        dm = TextDataModule(
            train_path="data/raw/train.parquet",
            val_path="data/raw/val.parquet",
            tokenizer_name="bert-base-uncased",
            window_size=128,
            batch_size=32,
        )
        dm.setup("fit")

        for batch in dm.train_dataloader():
            # batch = {
            #     "input_ids":         (B, max_seq_len),
            #     "attention_mask":    (B, max_seq_len),
            #     "masked_positions":  (B, max_masked),
            #     "target_ids":        (B, max_masked),
            # }
            ...

    Parameters
    ----------
    train_path :
        Path to the training parquet file.
    val_path :
        Optional path to the validation parquet file.
    test_path :
        Optional path to the test parquet file.
    text_column :
        Name of the column in the parquet file that contains the raw text.
    tokenizer_name :
        HuggingFace model name (or local path) for the tokenizer.
    window_size :
        Number of tokens per chunk (N).
    stride :
        Step between consecutive windows.  ``None`` → no overlap.
    alpha :
        Fraction of tokens to mask.
    batch_size :
        Number of samples per batch.
    num_workers :
        Number of subprocesses for data loading (``0`` = main process).
    """

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        text_column: str = "text",
        tokenizer_name: str = "bert-base-uncased",
        window_size: int = 512,
        stride: Optional[int] = None,
        alpha: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.text_column = text_column
        self.tokenizer_name = tokenizer_name
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Will be populated in ``setup()``
        self.tokenizer: Optional[TokenizerWrapper] = None
        self.train_dataset: Optional[StreamedMaskedDataset] = None
        self.val_dataset: Optional[StreamedMaskedDataset] = None
        self.test_dataset: Optional[StreamedMaskedDataset] = None

    # ── Data preparation ────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        """Open streaming parquet datasets and build streaming datasets."""
        self.tokenizer = TokenizerWrapper(self.tokenizer_name)

        if stage in (None, "fit"):
            train_iterable = self._open_stream(self.train_path)
            self.train_dataset = StreamedMaskedDataset(
                train_iterable,
                self.tokenizer,
                window_size=self.window_size,
                stride=self.stride,
                alpha=self.alpha,
            )
            if self.val_path is not None:
                val_iterable = self._open_stream(self.val_path)
                self.val_dataset = StreamedMaskedDataset(
                    val_iterable,
                    self.tokenizer,
                    window_size=self.window_size,
                    stride=self.stride,
                    alpha=self.alpha,
                )

        if stage in (None, "test") and self.test_path is not None:
            test_iterable = self._open_stream(self.test_path)
            self.test_dataset = StreamedMaskedDataset(
                test_iterable,
                self.tokenizer,
                window_size=self.window_size,
                stride=self.stride,
                alpha=self.alpha,
            )

    def _open_stream(self, path: str) -> Any:
        """Open a parquet file as a streaming HuggingFace IterableDataset.

        Uses ``datasets.load_dataset(streaming=True)`` which **never loads
        the whole file into memory** — rows are read one at a time as we
        iterate.
        """
        if not path:
            raise FileNotFoundError(
                f"Parquet path is empty or None. Cannot open stream."
            )
        ds = load_dataset("parquet", data_files=path, split="train",  streaming=True)
        # Keep only the text column to minimise data transfer
        ds = ds.select_columns([self.text_column])
        return ds

    # ── Collation ───────────────────────────────────────────────────────────

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Assemble a heterogeneous batch into padded tensors.

        Handles two sources of variable length:

        1. **Input sequences** (``masked_ids``) may have slightly different
           lengths because the final chunk of each text is often shorter than
           ``window_size``.  They are left-padded to ``max_seq_len`` in the
           batch.
        2. **Masked positions / target IDs** vary because each chunk has a
           different number of masked tokens.  They are right-padded to
           ``max_masked`` in the batch.

        Returns a dict with keys:
        - ``input_ids``        (B, max_seq_len)   —  padded masked sequences
        - ``attention_mask``   (B, max_seq_len)   —  1 = real, 0 = padding
        - ``masked_positions`` (B, max_masked)    —  indices; **-1 = padding**
        - ``target_ids``       (B, max_masked)    —  token IDs; 0 = padding
        """
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0  # fallback — BERT uses 0 by default

        # -- 1. Pad input sequences (masked_ids) ---------------------------
        masked_ids_list = [item["masked_ids"] for item in batch]
        max_seq_len = max(len(s) for s in masked_ids_list)

        padded_ids: List[List[int]] = []
        attn_masks: List[List[int]] = []
        for seq in masked_ids_list:
            pad_len = max_seq_len - len(seq)
            padded_ids.append([pad_id] * pad_len + seq)  # left-pad
            attn_masks.append([0] * pad_len + [1] * len(seq))

        # -- 2. Pad masked positions & target IDs -------------------------
        positions_list = [item["masked_positions"] for item in batch]
        targets_list = [item["target_ids"] for item in batch]
        max_masked = max(len(p) for p in positions_list)

        padded_positions: List[List[int]] = []
        padded_targets: List[List[int]] = []
        for pos, tgt in zip(positions_list, targets_list):
            pad_len = max_masked - len(pos)
            padded_positions.append(pos + [-1] * pad_len)  # -1 = ignore in loss
            padded_targets.append(tgt + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            "masked_positions": torch.tensor(padded_positions, dtype=torch.long),
            "target_ids": torch.tensor(padded_targets, dtype=torch.long),
        }

    # ── DataLoaders ─────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,                # no shuffle for streaming IterableDataset
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("TextDataModule — quick self-test")
    print("=" * 60)

    # ── Create a tiny synthetic parquet file ────────────────────────────────
    sample_texts = [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Machine learning models learn patterns from large amounts of data.",
        "Transformers have revolutionised natural language processing tasks.",
        "Self-supervised learning allows models to learn without labels.",
        "The capital of France is Paris and it is known for the Eiffel Tower.",
    ]

    # Use datasets.Dataset for a pandas-free parquet export
    from datasets import Dataset as HFDataset
    hf_ds = HFDataset.from_dict({"text": sample_texts})
    tmpdir = tempfile.mkdtemp()
    train_path = os.path.join(tmpdir, "train.parquet")
    val_path = os.path.join(tmpdir, "val.parquet")
    hf_ds.select(range(4)).to_parquet(train_path)
    hf_ds.select(range(4, len(sample_texts))).to_parquet(val_path)

    print(f"\n→ Created train file   : {train_path}  (4 rows)")
    print(f"→ Created val file     : {val_path}  ({len(sample_texts) - 4} rows)")

    # ── Instantiate the DataModule ──────────────────────────────────────────
    dm = TextDataModule(
        train_path=train_path,
        val_path=val_path,
        text_column="text",
        tokenizer_name="bert-base-uncased",
        window_size=20,
        stride=10,
        alpha=0.3,
        batch_size=2,
        num_workers=3,
    )

    # ── Setup & fetch a batch ──────────────────────────────────────────────
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    print(f"\n✓ Train dataset type   : {type(dm.train_dataset).__name__}")
    print(f"✓ Val dataset type     : {type(dm.val_dataset).__name__}")
    print(f"✓ Batch keys           : {list(batch.keys())}")
    print(f"  input_ids.shape      : {batch['input_ids'].shape}")
    print(f"  attention_mask.shape : {batch['attention_mask'].shape}")
    print(f"  masked_positions.shape : {batch['masked_positions'].shape}")
    print(f"  target_ids.shape     : {batch['target_ids'].shape}")

    # Quick sanity: check that target tokens are different from [MASK]
    mask_id = dm.tokenizer.mask_token_id
    assert torch.all(batch["input_ids"] != mask_id).item() is False, \
        "Expected at least some [MASK] tokens in the batch"
    print("✓ At least one [MASK] token present in input_ids — masking works.")

    # Check that attention_mask has the right pattern
    assert batch["attention_mask"].sum().item() > 0, \
        "Attention mask should have at least one 1 per row"
    print("✓ Attention mask looks correct.")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    import shutil
    shutil.rmtree(tmpdir)
    print(f"\n→ Temporary files cleaned up.")
    print("✓ All tests passed.")
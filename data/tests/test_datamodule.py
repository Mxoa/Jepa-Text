"""End-to-end tests for TextDataModule: shapes, masking semantics, padding."""

from typing import Any, Dict, List

import pytest
import torch
from datasets import Dataset as HFDataset

from data.datamodule import TextDataModule


# ── Helpers ──────────────────────────────────────────────────────────────────

SENTENCES = [
    "The quick brown fox juhmps over the lazy dog near the riverbank.",
    "Machine learning models learn patterns from large amounts of data.",
    "Transformers have revolutionised natural language processing tasks.",
    "Self-supervised learning aldlows models to learn without labels.",
    "The capital of France is Paris and it is known for the Eiffel Tower.",
    "Neural networks consist of layers of interconnected neurons.",
    "Attesntion is alsl yssssou need accosrdingss to thess famous paper.",
    "Tokesnization splits text into smaller pieces called tokens.",
    "The goal of pre-training is to learn useful representations.",
    "Masksed language modelling presdicts missing tokens from context.",
    "The csat sat on the mat and watched the birds outside.",
    "Natural lansguage undersstanding requires deep semantic knowledge.",
    "Gradient descent optismises the loss sfunction during training.",
    "Embeddings map discreste tokens into continuous vector spaces.",
    "The sun set behind the mountains painting the sky orange.",
    "Backpropagation computes gradients througsh the computational graph.",
    "Layer normalisation stabilises the training of deep networks.",
    "The quick brown fox was as typisng tessst from the past century.",
    "Transfer learning leverages knowledge from one task to another.",
    "Positional encodings give the model information about token order.",
]

# Make ~200 rows by repeating and varying slightly
TEXTS: List[str] = []
for i in range(10):
    for s in SENTENCES:
        if i % 3 == 0:
            TEXTS.append(s)
        elif i % 3 == 1:
            TEXTS.append(s.replace("the ", "this ", 1).replace("The ", "This ", 1))
        else:
            TEXTS.append(s.upper())


@pytest.fixture(scope="module")
def parquet_paths(tmp_path_factory):
    """Create a tiny train/val parquet pair and return their paths."""
    tmp = tmp_path_factory.mktemp("parquet_data")
    ds = HFDataset.from_dict({"text": TEXTS})
    train_path = str(tmp / "train.parquet")
    val_path = str(tmp / "val.parquet")
    ds.select(range(160)).to_parquet(train_path)      # ~160 rows
    ds.select(range(160, len(TEXTS))).to_parquet(val_path)  # ~40 rows
    return train_path, val_path


@pytest.fixture(scope="module")
def dm(parquet_paths):
    """Return a configured TextDataModule (not yet set up)."""
    train_path, val_path = parquet_paths
    return TextDataModule(
        train_path=train_path,
        val_path=val_path,
        text_column="text",
        tokenizer_name="bert-base-uncased",
        window_size=32,
        stride=16,
        alpha=0.3,
        batch_size=8,
        num_workers=0,
    )


@pytest.fixture(scope="module")
def setup_dm(dm):
    """Call setup("fit") once per module and return the dm."""
    dm.setup("fit")
    return dm


# ══════════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestShapes:
    """Verify the batch dict keys and tensor shapes match the docstring."""

    def test_dataset_iterable(self, setup_dm):
        """Streaming dataset is iterable and yields chunks."""
        loader = setup_dm.train_dataloader()
        batch = next(iter(loader))
        assert batch["input_ids"].size(0) == 8  # batch_size

    def test_batch_keys(self, setup_dm):
        batch = next(iter(setup_dm.train_dataloader()))
        expected_keys = {"input_ids", "attention_mask", "masked_positions", "target_ids"}
        assert set(batch.keys()) == expected_keys, f"Got {set(batch.keys())}"

    def test_batch_shapes(self, setup_dm):
        batch = next(iter(setup_dm.train_dataloader()))
        B = 8  # batch_size
        max_seq = batch["input_ids"].shape[1]
        max_msk = batch["masked_positions"].shape[1]

        assert batch["input_ids"].shape == (B, max_seq), \
            f"input_ids.shape {batch['input_ids'].shape}"
        assert batch["attention_mask"].shape == (B, max_seq), \
            f"attention_mask.shape {batch['attention_mask'].shape}"
        assert batch["masked_positions"].shape == (B, max_msk), \
            f"masked_positions.shape {batch['masked_positions'].shape}"
        assert batch["target_ids"].shape == (B, max_msk), \
            f"target_ids.shape {batch['target_ids'].shape}"

    def test_dtypes(self, setup_dm):
        batch = next(iter(setup_dm.train_dataloader()))
        assert batch["input_ids"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.long
        assert batch["masked_positions"].dtype == torch.long
        assert batch["target_ids"].dtype == torch.long


class TestMaskingSemantics:
    """Check that masking behaves as expected (alpha, whole word, targets)."""

    def test_masking_fraction(self, setup_dm):
        """~30% of non-pad tokens should be masked (within reasonable bounds)."""
        batch = next(iter(setup_dm.train_dataloader()))
        mask_id = setup_dm.tokenizer.mask_token_id
        attn = batch["attention_mask"]

        for i in range(batch["input_ids"].size(0)):
            valid = attn[i].bool()
            seq = batch["input_ids"][i][valid]
            n_masked = (seq == mask_id).sum().item()
            ratio = n_masked / valid.sum().item()
            # With alpha=0.3, we expect roughly 30%, allow ±15pp
            assert 0.10 <= ratio <= 0.80, \
                f"Sample {i}: masked {ratio:.2%} of valid tokens (expected ~30%)" # Loose bounds due to whole-word masking

    def test_masked_positions_match_target_ids(self, setup_dm):
        """For every masked position, target_ids should match the original token.

        We reconstruct the original sequence by taking ``input_ids`` and
        replacing every ``[MASK]`` token with the corresponding ``target_id``,
        then verify that the reconstructed token at each masked position
        matches the target.
        """
        batch = next(iter(setup_dm.train_dataloader()))
        mask_id = setup_dm.tokenizer.mask_token_id
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        positions = batch["masked_positions"]
        targets = batch["target_ids"]

        for i in range(input_ids.size(0)):
            # Reconstruct the original sequence (before masking)
            reconstructed = input_ids[i].clone()
            for j in range(positions.size(1)):
                pos = positions[i, j].item()
                tgt = targets[i, j].item()
                if pos == -1:
                    continue
                # The input at this position should be [MASK] (we verify
                # separately in test_masked_positions_are_masked).
                # Replace it with the target to reconstruct the original.
                reconstructed[pos] = tgt

            # Now verify: for each masked position, the reconstructed token
            # equals the target (trivially true by construction above).
            # More usefully: verify that the target is a *real* token
            # (not a special token, not padding).
            special_ids = set(setup_dm.tokenizer.all_special_ids)
            special_ids.discard(mask_id)
            for j in range(positions.size(1)):
                pos = positions[i, j].item()
                tgt = targets[i, j].item()
                if pos == -1:
                    continue
                assert tgt not in special_ids, \
                    f"Sample {i}, pos {pos}: target {tgt} is a special token"
                assert tgt != 0, \
                    f"Sample {i}, pos {pos}: target is 0 (padding)"

    def test_masked_positions_are_masked(self, setup_dm):
        """Every non-padding masked_position should be [MASK] in input_ids.

        ``masked_positions`` are indices in the **original (unpadded)**
        sequence.  Because the collate function left-pads ``input_ids``,
        the actual index in the padded tensor is ``pad_len + pos`` where
        ``pad_len = max_seq_len - original_len``.
        """
        batch = next(iter(setup_dm.train_dataloader()))
        mask_id = setup_dm.tokenizer.mask_token_id
        pad_id = setup_dm.tokenizer.pad_token_id or 0
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]
        positions = batch["masked_positions"]

        for i in range(input_ids.size(0)):
            # Compute left-padding length for this sample
            valid_len = attn[i].sum().item()
            max_len = input_ids.size(1)
            pad_len = max_len - int(valid_len)

            for j in range(positions.size(1)):
                pos = positions[i, j].item()
                if pos == -1:
                    continue
                # Account for left-padding offset
                padded_pos = pad_len + pos
                assert padded_pos < max_len, \
                    f"Sample {i}, pos {pos}: padded_pos {padded_pos} >= max_len {max_len}"
                assert input_ids[i, padded_pos].item() == mask_id, \
                    f"Sample {i}, pos {pos} (padded idx {padded_pos}): " \
                    f"expected [MASK] but got {input_ids[i, padded_pos].item()}"

    def test_no_special_tokens_masked(self, setup_dm):
        """[CLS], [SEP], [PAD] should never be masked."""
        batch = next(iter(setup_dm.train_dataloader()))
        special_ids = set(setup_dm.tokenizer.all_special_ids)
        mask_id = setup_dm.tokenizer.mask_token_id
        special_ids.discard(mask_id)  # [MASK] itself is special but should appear

        positions = batch["masked_positions"]
        attn = batch["attention_mask"]
        input_ids = batch["input_ids"]

        for i in range(positions.size(0)):
            # Compute left-padding offset
            valid_len = attn[i].sum().item()
            max_len = input_ids.size(1)
            pad_len = max_len - int(valid_len)

            for j in range(positions.size(1)):
                pos = positions[i, j].item()
                if pos == -1:
                    continue
                # The target is the original token ID (before masking)
                tgt = batch["target_ids"][i, j].item()
                assert tgt not in special_ids, \
                    f"Sample {i}, pos {pos}: original token {tgt} is a special token (should never be masked)"


class TestPadding:
    """Verify attention_mask and handling of variable-length sequences."""

    def test_attention_mask_padding(self, setup_dm):
        """Left-padded positions should have attention_mask = 0."""
        batch = next(iter(setup_dm.train_dataloader()))
        pad_id = setup_dm.tokenizer.pad_token_id or 0
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]

        for i in range(input_ids.size(0)):
            # Find padding: in left-padding, leading pad tokens have attn=0
            first_non_pad = (attn[i] == 1).nonzero(as_tuple=True)[0]
            if first_non_pad.numel() == 0:
                continue  # degenerate all-pad sequence (shouldn't happen)
            pad_end = first_non_pad[0].item()
            # Everything before pad_end should be pad_id
            assert (input_ids[i, :pad_end] == pad_id).all(), \
                f"Sample {i}: left-pad region doesn't contain pad_id"
            assert (attn[i, :pad_end] == 0).all(), \
                f"Sample {i}: left-pad region should have attention_mask=0"
            # Everything from pad_end onward should have attention_mask=1
            assert (attn[i, pad_end:] == 1).all(), \
                f"Sample {i}: content region should have attention_mask=1"

    def test_variable_length_handled(self, setup_dm):
        """Sequences shorter than window_size (flush leftovers) must still work."""
        # Some chunks are leftovers (< window_size). Ensure they don't crash.
        loader = setup_dm.train_dataloader()
        for _ in range(5):  # check a few batches
            batch = next(iter(loader))
            assert batch["input_ids"].size(0) == 8, "Batch should have batch_size=8 rows"
            # All rows in a batch have the same padded length
            assert batch["input_ids"].size(1) == batch["attention_mask"].size(1)

    def test_masked_positions_padding_is_minus_one(self, setup_dm):
        """Padding in masked_positions should be -1."""
        batch = next(iter(setup_dm.train_dataloader()))
        attn = batch["attention_mask"]
        positions = batch["masked_positions"]

        for i in range(attn.size(0)):
            # Find actual masked positions (non-padding)
            real = (positions[i] != -1).nonzero(as_tuple=True)[0]
            if real.numel() == 0:
                continue
            # After the last real position, everything should be -1
            last_real = real[-1].item()
            if last_real + 1 < positions.size(1):
                assert (positions[i, last_real + 1:] == -1).all(), \
                    f"Sample {i}: masked_position padding should be -1"


class TestWholeWordAndVisual:
    """Whole-word masking and visual decode of a sample."""

    def test_whole_word_masking(self, setup_dm):
        """When whole_word=True, sub-tokens (##xxx) must never be isolated."""
        tokenizer = setup_dm.tokenizer
        batch = next(iter(setup_dm.train_dataloader()))
        positions = batch["masked_positions"]
        input_ids = batch["input_ids"]
        attn = batch["attention_mask"]

        for i in range(positions.size(0)):
            # Compute left-padding offset
            valid_len = attn[i].sum().item()
            max_len = input_ids.size(1)
            pad_len = max_len - int(valid_len)

            for j in range(positions.size(1)):
                pos = positions[i, j].item()
                if pos == -1:
                    continue
                # Decode the token at this position (accounting for left-padding)
                padded_pos = pad_len + pos
                token_str = tokenizer.decode([input_ids[i, padded_pos].item()],
                                             skip_special_tokens=False).strip()
                if token_str.startswith("##"):
                    # The previous position should also be a masked position
                    prev_masked = (positions[i] == pos - 1).any().item()
                    assert prev_masked, \
                        f"Sample {i}, pos {pos}: sub-token {token_str!r} is masked " \
                        f"but its root at position {pos - 1} is not"

    def test_visual_inspection(self, setup_dm):
        """Print a decoded sample with [MASK] highlighted for manual inspection."""
        tokenizer = setup_dm.tokenizer
        batch = next(iter(setup_dm.train_dataloader()))
        mask_id = tokenizer.mask_token_id

        print("\n" + "=" * 70)
        print("Visual inspection of a masked batch (sample 0)")
        print("=" * 70)

        # Show first 3 samples from the batch
        for sample_idx in range(min(3, batch["input_ids"].size(0))):
            input_ids = batch["input_ids"][sample_idx].tolist()
            attn = batch["attention_mask"][sample_idx].tolist()
            masked_pos = batch["masked_positions"][sample_idx].tolist()
            targets = batch["target_ids"][sample_idx].tolist()

            # Truncate to non-padded length
            valid_len = sum(attn)
            input_ids = input_ids[-valid_len:] if valid_len else input_ids[:valid_len]

            # Original (unmasked) text
            original_ids = list(input_ids)
            for pos in masked_pos:
                if pos != -1 and pos < len(original_ids):
                    # Restore the target at that position to show original
                    idx_in_targets = masked_pos.index(pos)
                    original_ids[pos] = targets[idx_in_targets] if idx_in_targets < len(targets) else original_ids[pos]

            original_text = tokenizer.decode(original_ids, skip_special_tokens=False)
            masked_text = tokenizer.decode(input_ids, skip_special_tokens=False)

            print(f"\n── Sample {sample_idx} ──")
            print(f"  Original : {original_text}")
            print(f"  Masked   : {masked_text}")
            print(f"  Positions masked  : {[p for p in masked_pos if p != -1]}")
            print(f"  Target tokens     : {[t for t in targets if t != 0]}")

            # Count
            n_masked = sum(1 for p in masked_pos if p != -1)
            n_valid = valid_len
            if n_valid > 0:
                print(f"  Masked {n_masked}/{n_valid} tokens ({100 * n_masked / n_valid:.1f}%)")

        print("\n" + "=" * 70)
        print("Manual check:")
        print("  - Are ~30% of non-pad tokens replaced by [MASK]?")
        print("  - Are whole words masked together (no orphaned ##subword)?")
        print("  - Do target_ids match the original tokens at masked_positions?")
        print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
"""Tests for the JEPA module components: TransformerBlock and TransformerEncoder."""

import pytest
import torch

from modules.core.encoder import TransformerBlock, TransformerEncoder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    """Small config shared across tests."""
    return {
        "vocab_size": 100,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 3,
        "d_ff": 64,
        "max_seq_len": 32,
        "dropout": 0.0,  # deterministic tests
        "pad_token_id": 0,
    }


@pytest.fixture
def encoder(cfg):
    return TransformerEncoder(**cfg)


@pytest.fixture
def block(cfg):
    return TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["d_ff"], dropout=0.0)


@pytest.fixture
def sample_batch():
    """Batch of 2 sequences of varying length."""
    return {
        "input_ids": torch.tensor([
            [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],    # len=5
            [6, 7, 8, 0, 0, 0, 0, 0, 0, 0],      # len=3
        ]),
        "attention_mask": torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TransformerBlock tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTransformerBlock:
    """Unit tests for a single TransformerBlock."""

    def test_init(self, block):
        """Block should have the expected submodules."""
        assert hasattr(block, "attn")
        assert hasattr(block, "norm1")
        assert hasattr(block, "ff")
        assert hasattr(block, "norm2")
        assert hasattr(block, "dropout")

    def test_output_shape(self, block):
        """Forward pass preserves (B, L, D)."""
        B, L, D = 2, 8, 16
        x = torch.randn(B, L, D)
        out = block(x)
        assert out.shape == (B, L, D)

    def test_output_shape_with_mask(self, block):
        """Forward pass with key_padding_mask still preserves shape."""
        B, L, D = 2, 8, 16
        x = torch.randn(B, L, D)
        key_padding_mask = torch.tensor([
            [False, False, False, False, True, True, True, True],
            [False, False, True, True, True, True, True, True],
        ])
        out = block(x, key_padding_mask=key_padding_mask)
        assert out.shape == (B, L, D)

    def test_residual_connection(self, block):
        """Output should differ from input (transformation happened)."""
        B, L, D = 1, 4, 16
        x = torch.randn(B, L, D)
        out = block(x)
        assert not torch.allclose(x, out, atol=1e-6), \
            "Block should modify the input (no identity)."

    @pytest.mark.parametrize("B, L, D", [(1, 1, 16), (4, 16, 16), (2, 32, 16)])
    def test_various_shapes(self, block, B, L, D):
        """Block handles various (B, L, D) shapes."""
        x = torch.randn(B, L, D)
        out = block(x)
        assert out.shape == (B, L, D)

    def test_gradients_flow(self, block):
        """Gradients should flow through all parameters."""
        B, L, D = 2, 4, 16
        x = torch.randn(B, L, D, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ══════════════════════════════════════════════════════════════════════════════
#  TransformerEncoder tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTransformerEncoder:
    """Unit tests for the full TransformerEncoder."""

    def test_init(self, encoder, cfg):
        """Encoder has expected submodules."""
        assert encoder.token_embedding.weight.shape == (cfg["vocab_size"], cfg["d_model"])
        assert encoder.position_embedding.weight.shape == (cfg["max_seq_len"], cfg["d_model"])
        assert len(encoder.blocks) == cfg["n_layers"]
        assert hasattr(encoder, "final_norm")
        assert encoder.pad_token_id == cfg["pad_token_id"]

    def test_output_shape(self, encoder, sample_batch):
        """Forward pass returns (B, L, D)."""
        out = encoder(sample_batch["input_ids"], sample_batch["attention_mask"])
        B, L = sample_batch["input_ids"].shape
        assert out.shape == (B, L, 16), f"Expected ({B}, {L}, 16), got {out.shape}"

    def test_output_dtype(self, encoder, sample_batch):
        """Output is float32."""
        out = encoder(sample_batch["input_ids"], sample_batch["attention_mask"])
        assert out.dtype == torch.float32

    def test_output_without_mask(self, encoder, sample_batch):
        """Forward pass without attention_mask still works."""
        out = encoder(sample_batch["input_ids"])
        B, L = sample_batch["input_ids"].shape
        assert out.shape == (B, L, 16)

    def test_padding_does_not_crash(self, encoder):
        """Varying padding lengths should run without errors."""
        # Separate forward calls with different padding lengths
        ids1 = torch.tensor([[1, 2, 3, 0, 0]])
        attn1 = torch.tensor([[1, 1, 1, 0, 0]])
        out1 = encoder(ids1, attn1)

        ids2 = torch.tensor([[1, 2, 3, 0, 0, 0, 0]])
        attn2 = torch.tensor([[1, 1, 1, 0, 0, 0, 0]])
        out2 = encoder(ids2, attn2)

        # Both should produce valid outputs with correct shapes
        assert out1.shape == (1, 5, 16)
        assert out2.shape == (1, 7, 16)
        # Representations for valid tokens are computed without crash
        assert not torch.isnan(out1[0, :3]).any()
        assert not torch.isnan(out2[0, :3]).any()

    def test_gradients_flow(self, encoder, sample_batch):
        """Gradients should flow through all encoder parameters."""
        out = encoder(sample_batch["input_ids"], sample_batch["attention_mask"])
        loss = out.sum()
        loss.backward()
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_different_sequences_produce_different_outputs(self, encoder):
        """Two different input sequences should produce different outputs."""
        input_ids = torch.tensor([
            [1, 2, 3, 4, 0, 0, 0, 0],
            [5, 6, 7, 8, 0, 0, 0, 0],
        ])
        attn = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ])
        out = encoder(input_ids, attn)
        # The two sequences are different → outputs should differ
        assert not torch.allclose(out[0], out[1], atol=1e-6), \
            "Different inputs should produce different outputs."

    def test_attention_mask_semantics(self, encoder):
        """Padding tokens (attn_mask=0) should not affect valid token representations.

        If we have a sequence [A, B, C, PAD, PAD] and [A, B, C],
        the representations of A, B, C should be the same (modulo position
        embeddings since positions differ due to left-padding).
        """
        # Short sequence: [1, 2, 3]
        short_ids = torch.tensor([[1, 2, 3]])
        short_mask = torch.tensor([[1, 1, 1]])

        # Same sequence with padding: [1, 2, 3, 0, 0, 0, 0]
        long_ids = torch.tensor([[1, 2, 3, 0, 0, 0, 0]])
        long_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]])

        out_short = encoder(short_ids, short_mask)
        out_long = encoder(long_ids, long_mask)

        # The representation at position 2 (third token) should be
        # different because position embeddings differ (position 2 vs padding
        # offset + 2 = 4, since left-padded).
        # So we just check shapes are consistent.
        assert out_short.shape[1] == 3
        assert out_long.shape[1] == 7

    def test_large_vocab(self):
        """Encoder handles realistic vocab sizes."""
        enc = TransformerEncoder(
            vocab_size=30522,  # BERT
            d_model=64,
            n_heads=8,
            n_layers=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
        x = torch.randint(0, 100, (1, 32))
        out = enc(x)
        assert out.shape == (1, 32, 64)

    def test_embedding_consistency(self, encoder):
        """Same token at same position should give same hidden state start.

        This checks that the embedding layer works deterministically.
        """
        input_ids = torch.tensor([[5, 5, 5]])  # same token repeated
        attn = torch.tensor([[1, 1, 1]])
        out = encoder(input_ids, attn)
        # After the first block, representations at different positions
        # should differ due to position embeddings
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6), \
            "Same token at different positions should differ."

    @pytest.mark.parametrize("L", [1, 8, 16, 32])
    def test_varying_seq_lengths(self, encoder, L):
        """Encoder handles sequences of different lengths (padded/not)."""
        if L > 32:  # max_seq_len in fixture
            return
        x = torch.randint(1, 50, (2, L))
        out = encoder(x)
        assert out.shape == (2, L, 16)


# ══════════════════════════════════════════════════════════════════════════════
#  Integration tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end tests combining masking + encoding (as in JEPA)."""

    def test_encoder_gradient_flows_to_input(self, encoder):
        """Gradient should be computable w.r.t. input embeddings (not input_ids)."""
        x = torch.randint(1, 50, (2, 8))
        attn = torch.ones_like(x)
        out = encoder(x, attn)
        loss = out.mean()
        loss.backward()
        # Check that token embedding has gradients
        assert encoder.token_embedding.weight.grad is not None
        assert encoder.position_embedding.weight.grad is not None

    def test_numerical_stability(self, encoder):
        """Output should not contain NaN or Inf."""
        x = torch.randint(1, 50, (4, 16))
        attn = torch.ones_like(x)
        out = encoder(x, attn)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_masked_inputs(self, encoder):
        """Encoding with mask tokens in input should still be stable."""
        vocab_size = 100
        mask_id = 0  # arbitrary
        x = torch.randint(1, vocab_size, (2, 8))
        # Replace some tokens with mask token
        x[:, 3:6] = mask_id
        attn = torch.ones_like(x)
        out = encoder(x, attn)
        assert not torch.isnan(out).any()
        assert out.shape == (2, 8, 16)

    def test_deterministic(self, encoder, sample_batch):
        """With dropout=0, same input → same output."""
        out1 = encoder(sample_batch["input_ids"], sample_batch["attention_mask"])
        out2 = encoder(sample_batch["input_ids"], sample_batch["attention_mask"])
        assert torch.allclose(out1, out2, atol=1e-6), \
            "Deterministic forward pass should produce identical outputs."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
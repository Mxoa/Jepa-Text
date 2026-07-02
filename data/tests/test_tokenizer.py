"""Tests for the TokenizerWrapper class."""

from typing import List

import pytest

from data.tokenizer import TokenizerWrapper


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tokenizer() -> TokenizerWrapper:
    """Return a TokenizerWrapper with the default BERT uncased tokenizer."""
    return TokenizerWrapper()


@pytest.fixture(scope="module")
def gpt2_tokenizer() -> TokenizerWrapper:
    """Return a TokenizerWrapper with GPT-2 (no padding token by default)."""
    return TokenizerWrapper("gpt2")


# ── Initialisation ────────────────────────────────────────────────────────


class TestInit:
    def test_default_model(self) -> None:
        tok = TokenizerWrapper()
        assert tok.tokenizer.name_or_path == "bert-base-uncased"
        assert tok.max_length == 512
        assert tok.truncation is True
        assert tok.padding is False
        assert tok.add_special_tokens is True

    def test_custom_model(self) -> None:
        tok = TokenizerWrapper("gpt2", max_length=128, padding=True)
        assert tok.tokenizer.name_or_path == "gpt2"
        assert tok.max_length == 128
        assert tok.padding is True

    def test_repr(self, tokenizer: TokenizerWrapper) -> None:
        r = repr(tokenizer)
        assert "TokenizerWrapper" in r
        assert "bert-base-uncased" in r

    def test_len(self, tokenizer: TokenizerWrapper) -> None:
        assert len(tokenizer) == tokenizer.vocab_size > 0


# ── Encoding ──────────────────────────────────────────────────────────────


class TestEncode:
    def test_encode_single_text(self, tokenizer: TokenizerWrapper) -> None:
        ids = tokenizer.encode("Hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_with_special_tokens(self, tokenizer: TokenizerWrapper) -> None:
        ids = tokenizer.encode("Hello world", add_special_tokens=True)
        # BERT adds [CLS] at start and [SEP] at end
        assert ids[0] == tokenizer.cls_token_id
        assert ids[-1] == tokenizer.sep_token_id

    def test_encode_without_special_tokens(self, tokenizer: TokenizerWrapper) -> None:
        ids = tokenizer.encode("Hello world", add_special_tokens=False)
        assert ids[0] != tokenizer.cls_token_id
        assert ids[-1] != tokenizer.sep_token_id

    def test_encode_truncation(self, tokenizer: TokenizerWrapper) -> None:
        long_text = "word " * 1000
        ids = tokenizer.encode(long_text, max_length=10, truncation=True)
        assert len(ids) <= 10

    def test_encode_padding(self, tokenizer: TokenizerWrapper) -> None:
        ids = tokenizer.encode("Hello", max_length=10, padding="max_length")
        assert len(ids) == 10
        # Padded positions should be pad_token_id
        assert ids[-1] == tokenizer.pad_token_id

    def test_encode_empty_string(self, tokenizer: TokenizerWrapper) -> None:
        ids = tokenizer.encode("")
        assert isinstance(ids, list)
        assert len(ids) > 0  # at least special tokens


class TestEncodeBatch:
    def test_encode_batch(self, tokenizer: TokenizerWrapper) -> None:
        texts = ["Hello world", "How are you?"]
        out = tokenizer.encode_batch(texts)
        assert "input_ids" in out
        assert "attention_mask" in out
        assert len(out["input_ids"]) == 2
        assert len(out["attention_mask"]) == 2

    def test_encode_batch_with_tensors(self, tokenizer: TokenizerWrapper) -> None:
        texts = ["Hello world", "How are you?"]
        # padding + truncation required so all sequences have the same length
        out = tokenizer.encode_batch(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        assert hasattr(out["input_ids"], "shape")  # is a tensor

    def test_encode_batch_padding(self, tokenizer: TokenizerWrapper) -> None:
        texts = ["Hi", "A longer sentence here for testing"]
        out = tokenizer.encode_batch(texts, padding=True)
        # All sequences should have the same length after padding
        lengths = [len(ids) for ids in out["input_ids"]]
        assert all(l == lengths[0] for l in lengths)


# ── Decoding ──────────────────────────────────────────────────────────────


class TestDecode:
    def test_decode_roundtrip(self, tokenizer: TokenizerWrapper) -> None:
        original = "Hello world"
        ids = tokenizer.encode(original, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded.strip().lower() == original.lower()

    def test_decode_single_id(self, tokenizer: TokenizerWrapper) -> None:
        token = tokenizer.decode([tokenizer.cls_token_id], skip_special_tokens=False)
        assert tokenizer.cls_token in token

    def test_decode_skip_special(self, tokenizer: TokenizerWrapper) -> None:
        ids = tokenizer.encode("test", add_special_tokens=True)
        decoded_with = tokenizer.decode(ids, skip_special_tokens=False)
        decoded_without = tokenizer.decode(ids, skip_special_tokens=True)
        assert len(decoded_with) >= len(decoded_without)

    def test_decode_batch(self, tokenizer: TokenizerWrapper) -> None:
        texts = ["Hello world", "How are you?"]
        batch_ids = [
            tokenizer.encode(t, add_special_tokens=False) for t in texts
        ]
        decoded = tokenizer.decode_batch(batch_ids)
        assert len(decoded) == 2
        assert decoded[0].strip().lower() == texts[0].lower()


# ── Properties ────────────────────────────────────────────────────────────


class TestProperties:
    def test_vocab_size(self, tokenizer: TokenizerWrapper) -> None:
        assert tokenizer.vocab_size > 0

    def test_special_token_ids(self, tokenizer: TokenizerWrapper) -> None:
        assert tokenizer.cls_token_id is not None
        assert tokenizer.sep_token_id is not None
        assert tokenizer.pad_token_id is not None
        assert tokenizer.mask_token_id is not None
        assert tokenizer.unk_token_id is not None

    def test_special_tokens(self, tokenizer: TokenizerWrapper) -> None:
        assert tokenizer.cls_token == "[CLS]"
        assert tokenizer.sep_token == "[SEP]"
        assert tokenizer.pad_token == "[PAD]"
        assert tokenizer.mask_token == "[MASK]"
        assert tokenizer.unk_token == "[UNK]"

    def test_all_special(self, tokenizer: TokenizerWrapper) -> None:
        assert len(tokenizer.all_special_ids) > 0
        assert len(tokenizer.all_special_tokens) > 0
        assert len(tokenizer.all_special_ids) == len(tokenizer.all_special_tokens)

    def test_model_max_length(self, tokenizer: TokenizerWrapper) -> None:
        assert tokenizer.model_max_length > 0

    def test_gpt2_no_pad_token(self, gpt2_tokenizer: TokenizerWrapper) -> None:
        # GPT-2 doesn't have a pad token by default
        assert gpt2_tokenizer.pad_token_id is None
        assert gpt2_tokenizer.pad_token is None


# ── Utility methods ───────────────────────────────────────────────────────


class TestUtils:
    def test_num_tokens(self, tokenizer: TokenizerWrapper) -> None:
        n = tokenizer.num_tokens("Hello world")
        assert n > 0
        assert isinstance(n, int)

    def test_convert_ids_to_tokens(self, tokenizer: TokenizerWrapper) -> None:
        tokens = tokenizer.convert_ids_to_tokens([tokenizer.cls_token_id])
        assert tokens == ["[CLS]"]

    def test_convert_tokens_to_ids(self, tokenizer: TokenizerWrapper) -> None:
        ids = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
        assert ids == [tokenizer.cls_token_id, tokenizer.sep_token_id]

    def test_roundtrip_ids_tokens(self, tokenizer: TokenizerWrapper) -> None:
        ids = [2009, 2005, 2009]  # known BERT tokens
        tokens = tokenizer.convert_ids_to_tokens(ids)
        ids_back = tokenizer.convert_tokens_to_ids(tokens)
        assert ids_back == ids

    def test_save_and_load(self, tokenizer: TokenizerWrapper, tmp_path) -> None:
        save_dir = tmp_path / "saved_tokenizer"
        tokenizer.save_pretrained(str(save_dir))
        assert (save_dir / "tokenizer.json").exists()

        # Reload from saved path
        loaded = TokenizerWrapper(str(save_dir))
        assert loaded.vocab_size == tokenizer.vocab_size
        ids = loaded.encode("Hello world")
        assert len(ids) > 0
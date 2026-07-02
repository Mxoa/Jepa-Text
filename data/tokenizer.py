"""Tokenizer wrapper from hugging face"""

from typing import List, Optional, Union
from transformers import AutoTokenizer
import os




class TokenizerWrapper:
    """Wrapper around a Hugging Face tokenizer for convenient encoding/decoding."""

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        max_length: int = 512,
        truncation: bool = True,
        padding: Union[bool, str] = False,
        add_special_tokens: bool = True,
    ):
        """Initialize the tokenizer wrapper.

        Args:
            model_name_or_path: Name or path of the pretrained tokenizer.
            max_length: Maximum sequence length for truncation/padding.
            truncation: Whether to truncate sequences longer than max_length.
            padding: Whether to pad sequences to max_length.
            add_special_tokens: Whether to add special tokens (CLS, SEP, etc.).
        """

        safe_model_name = model_name_or_path.replace("/", "_")
        local_dir = os.path.join("./data/tokenizers", safe_model_name)

        # 2. Vérifier si le dossier existe et contient déjà un tokenizer valide
        if os.path.exists(local_dir) and os.path.isdir(local_dir):
            print(f"Chargement du tokenizer depuis le cache local : {local_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        else:
            print(f"Tokenizer non trouvé en local. Téléchargement depuis le Hub de Hugging Face : {model_name_or_path}")
            # Téléchargement depuis le Hub
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Création du dossier 'tokenizers/' et du sous-dossier s'ils n'existent pas
            os.makedirs(local_dir, exist_ok=True)
            
            # Sauvegarde locale pour la prochaine fois
            self.tokenizer.save_pretrained(local_dir)
            print(f"Tokenizer sauvegardé localement dans : {local_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        self.add_special_tokens = add_special_tokens

        # Add a [MASK] special token to the vocabulary
        self.add_mask_token()

    # ── Encoding ──────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        padding: Optional[Union[bool, str]] = None,
        add_special_tokens: Optional[bool] = None,
    ) -> List[int]:
        """Encode a single text into token IDs.

        Args:
            text: Input text to tokenize.
            max_length: Override default max_length.
            truncation: Override default truncation.
            padding: Override default padding.
            add_special_tokens: Override default add_special_tokens.

        Returns:
            List of token IDs.
        """
        return self.tokenizer.encode(
            text,
            max_length=max_length or self.max_length,
            truncation=truncation if truncation is not None else self.truncation,
            padding=padding if padding is not None else self.padding,
            add_special_tokens=(
                add_special_tokens
                if add_special_tokens is not None
                else self.add_special_tokens
            ),
        )

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        padding: Optional[Union[bool, str]] = None,
        add_special_tokens: Optional[bool] = None,
        return_tensors: Optional[str] = None,
    ) -> dict:
        """Encode a batch of texts, returning a dictionary of tensors/lists.

        Args:
            texts: List of input texts.
            max_length: Override default max_length.
            truncation: Override default truncation.
            padding: Override default padding.
            add_special_tokens: Override default add_special_tokens.
            return_tensors: Return format — None for lists, "pt" for PyTorch,
                            "tf" for TensorFlow, "np" for NumPy.

        Returns:
            Dictionary with keys "input_ids", "attention_mask", and optionally
            "token_type_ids".
        """
        return self.tokenizer(
            texts,
            max_length=max_length or self.max_length,
            truncation=truncation if truncation is not None else self.truncation,
            padding=padding if padding is not None else self.padding,
            add_special_tokens=(
                add_special_tokens
                if add_special_tokens is not None
                else self.add_special_tokens
            ),
            return_tensors=return_tensors,
        )

    # ── Decoding ──────────────────────────────────────────────────────────

    def decode(
        self,
        token_ids: Union[List[int], int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs back into a string.

        Args:
            token_ids: Single token ID or list of token IDs.
            skip_special_tokens: Whether to remove special tokens from output.
            clean_up_tokenization_spaces: Whether to clean up tokenization
                                          artifacts (e.g. extra spaces).

        Returns:
            Decoded string.
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def decode_batch(
        self,
        batch_token_ids: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """Decode a batch of token ID sequences into strings.

        Args:
            batch_token_ids: List of token ID sequences.
            skip_special_tokens: Whether to remove special tokens from output.
            clean_up_tokenization_spaces: Whether to clean up tokenization
                                          artifacts.

        Returns:
            List of decoded strings.
        """
        return self.tokenizer.batch_decode(
            batch_token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size of the tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> Optional[int]:
        """Return the pad token ID, or None if not set."""
        return self.tokenizer.pad_token_id

    @property
    def cls_token_id(self) -> Optional[int]:
        """Return the CLS token ID, or None if not set."""
        return self.tokenizer.cls_token_id

    @property
    def sep_token_id(self) -> Optional[int]:
        """Return the SEP token ID, or None if not set."""
        return self.tokenizer.sep_token_id

    @property
    def mask_token_id(self) -> Optional[int]:
        """Return the MASK token ID, or None if not set."""
        return self.tokenizer.mask_token_id

    @property
    def unk_token_id(self) -> Optional[int]:
        """Return the UNK token ID, or None if not set."""
        return self.tokenizer.unk_token_id

    @property
    def pad_token(self) -> Optional[str]:
        """Return the pad token string, or None if not set."""
        return self.tokenizer.pad_token

    @property
    def cls_token(self) -> Optional[str]:
        """Return the CLS token string, or None if not set."""
        return self.tokenizer.cls_token

    @property
    def sep_token(self) -> Optional[str]:
        """Return the SEP token string, or None if not set."""
        return self.tokenizer.sep_token

    @property
    def mask_token(self) -> Optional[str]:
        """Return the MASK token string, or None if not set."""
        return self.tokenizer.mask_token

    @property
    def unk_token(self) -> Optional[str]:
        """Return the UNK token string, or None if not set."""
        return self.tokenizer.unk_token

    @property
    def all_special_ids(self) -> List[int]:
        """Return the list of all special token IDs."""
        return self.tokenizer.all_special_ids

    @property
    def all_special_tokens(self) -> List[str]:
        """Return the list of all special token strings."""
        return self.tokenizer.all_special_tokens

    @property
    def model_max_length(self) -> int:
        """Return the maximum length the tokenizer was configured with."""
        return self.tokenizer.model_max_length

    # ── Utility methods ───────────────────────────────────────────────────

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a text (without padding)."""
        return len(self.encode(text, padding=False))

    def convert_ids_to_tokens(
        self, token_ids: Union[List[int], int]
    ) -> Union[List[str], str]:
        """Convert token ID(s) to the corresponding token string(s).

        Args:
            token_ids: Single token ID or list of token IDs.

        Returns:
            Corresponding token string(s).
        """
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def convert_tokens_to_ids(
        self, tokens: Union[List[str], str]
    ) -> Union[List[int], int]:
        """Convert token string(s) to the corresponding ID(s).

        Args:
            tokens: Single token string or list of token strings.

        Returns:
            Corresponding token ID(s).
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def add_mask_token(self, token: str = "[MASK]") -> int:
        """Add a [MASK] token to the tokenizer if it doesn't already have one.

        This is useful for models like GPT-2 that don't include a mask token
        by default. The token is added to the vocabulary and set as the
        tokenizer's mask token.

        Args:
            token: The mask token string to add (default "[MASK]").

        Returns:
            The ID of the newly added (or already existing) mask token.

        Raises:
            ValueError: If the tokenizer already has a different mask token
                        and the requested token doesn't match.
        """
        if self.tokenizer.mask_token is not None:
            if self.tokenizer.mask_token != token:
                raise ValueError(
                    f"Tokenizer already has mask_token={self.tokenizer.mask_token!r}, "
                    f"cannot set it to {token!r}. Use a different tokenizer."
                )
            return self.tokenizer.mask_token_id

        # Add the token to the vocabulary and set it as the mask token
        self.tokenizer.add_special_tokens({"mask_token": token})
        return self.tokenizer.mask_token_id

    def save_pretrained(self, save_directory: str) -> None:
        """Save the tokenizer to a directory.

        Args:
            save_directory: Path to the directory where the tokenizer
                            will be saved.
        """
        self.tokenizer.save_pretrained(save_directory)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.tokenizer.name_or_path!r}, "
            f"vocab_size={self.vocab_size}, "
            f"max_length={self.max_length})"
        )

    def __len__(self) -> int:
        """Return the vocabulary size (same as vocab_size)."""
        return self.vocab_size
    

if __name__ == "__main__":
    tokenizer = TokenizerWrapper()
    print(tokenizer.all_special_tokens)
    print(tokenizer.encode("without"))
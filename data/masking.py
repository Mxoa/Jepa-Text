"""Masking strategies for self-supervised learning on token sequences."""

import math
import random
import warnings
from typing import List, Optional, Tuple

from data.tokenizer import TokenizerWrapper


class Masking:
    """Apply masking strategies to token ID sequences.

    Two strategies are provided:

    - ``"random"`` : randomly mask ``alpha`` % of the tokens independently.
    - ``"block"``  : partition the tokens to mask into ``n_blocks`` contiguous
      blocks, each of (approximately) equal size, placed at random positions
      in the sequence (with wrap-around and skip-already-masked logic).

    In both cases the output is:
        - **masked_ids** : the input sequence with selected positions replaced
          by ``mask_token_id``.
        - **masked_positions** : the indices that were masked.
        - **target_ids** : the original token IDs that were at those positions
          (i.e. what the model should predict).

    Args:
        tokenizer: A ``TokenizerWrapper`` instance (needed to get the
            ``mask_token_id`` and optionally for whole-word alignment).
        alpha: Fraction of tokens to mask (between 0.0 and 1.0).
        n_blocks: Number of contiguous blocks (only used when
            ``strategy="block"``).  Default 1.
        strategy: ``"random"`` or ``"block"``. (block is not implemented yet, but would be nice to compare)
        whole_word: If ``True``, try to align masks on word boundaries
            (currently uses BERT-style ``##`` prefix detection).
    """

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        alpha: float = 0.15,
        n_blocks: int = 1,
        strategy: str = "random",
        whole_word: bool = True,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_blocks < 1:
            raise ValueError(f"n_blocksand not self.tokenizer.all_special_ids must be >= 1, got {n_blocks}")
        if strategy not in ("random", "block"):
            raise ValueError(f"strategy must be 'random' or 'block', got {strategy!r}")

        self.tokenizer = tokenizer
        self.alpha = alpha
        self.n_blocks = n_blocks
        self.strategy = strategy
        self.whole_word = whole_word

        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            # Try to add it (useful for GPT-2 etc.)
            mask_id = tokenizer.add_mask_token()
        self._mask_id = mask_id

    # ── Public API ────────────────────────────────────────────────────────

    def apply(self, token_ids: List[int]) -> dict:
        """Apply the configured masking strategy to *token_ids*.

        Returns:
            A dict with keys:
            - ``"masked_ids"``: list of token IDs with masks applied.
            - ``"masked_positions"``: list of masked indices.
            - ``"target_ids"``: original token IDs at those positions.
        """
        if self.strategy == "random":
            return self._mask_random(token_ids)
        else:
            raise NotImplementedError("Not Implemented, connot use another strategy for now.")

    # ── Random strategy ──────────────────────────────────────────────────

    def _mask_random(self, token_ids: List[int]) -> dict:
        n = len(token_ids)
        n_to_mask = max(1, int(round(self.alpha * n)))

        if n_to_mask >= n:
            warnings.warn(
                f"alpha={self.alpha} would mask {n_to_mask}/{n} tokens — "
                f"all tokens will be masked"
            )
            n_to_mask = n

        # Randomly choose positions without replacement
        positions = random.sample(range(n), n_to_mask)

        masked_ids = list(token_ids)
        target_ids = []
        masked_positions = []

        for pos in sorted(positions):
            if masked_ids[pos] not in self.tokenizer.all_special_ids: # On recouvre pas les caractères spéciaux
                masked_ids[pos] = self._mask_id
                target_ids.append(token_ids[pos])
                masked_positions.append(pos)

                if self.whole_word and pos < len(token_ids) - 1:
                    k = pos+1
                    next_token = self.tokenizer.decode([token_ids[k]])
                    while next_token.startswith("##") and k < len(token_ids):
                        next_token = self.tokenizer.decode([token_ids[k]])
                        masked_ids[k] = self._mask_id
                        target_ids.append(token_ids[k])
                        masked_positions.append(k)
                        k+=1

                    
                    k = pos
                    curr_token = self.tokenizer.decode([token_ids[k]])
                    back_fill = curr_token.startswith('##')
                    while k > 0 and back_fill:
                        k -= 1
                        prev_token = self.tokenizer.decode([token_ids[k]])
                        masked_ids[k] = self._mask_id
                        target_ids.append(token_ids[k])
                        masked_positions.append(k)
                        back_fill = prev_token.startswith('##')

        return {
            "masked_ids": masked_ids,
            "masked_positions": masked_positions,
            "target_ids": target_ids,
        }

    # ── Representation ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.strategy!r}, "
            f"alpha={self.alpha}, "
            f"n_blocks={self.n_blocks}, "
            f"whole_word={self.whole_word})"
        )
    
if __name__ == "__main__":
    from data.chunking import Chunking
    tokenizer = TokenizerWrapper()
    chunker = Chunking(tokenizer, 50, 1)
    masker = Masking(tokenizer, alpha=0.3, whole_word=True, strategy="random")

    for c in chunker.iterate(["Hello, i am from new orlean, a city located in USA", "bruh idk what you said please repea"]):

        msk = masker.apply(c)

        print(tokenizer.decode(msk['target_ids'], False), '--> (masking)', tokenizer.decode(msk['masked_ids'], False))

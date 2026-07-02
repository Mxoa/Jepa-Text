"""Sliding window chunking over tokenized text using a token buffer."""

from typing import Iterator, List, Optional

from data.tokenizer import TokenizerWrapper


class Chunking:
    """Chunk texts into fixed-size token windows via a token buffer.

    The workflow:
      1. Iterate over a list of texts.
      2. Tokenize one text at a time and append the token IDs to an internal buffer.
      3. As soon as the buffer contains at least ``window_size`` tokens,
         yield chunks of exactly ``window_size`` consecutive tokens (sliding
         the buffer forward by ``stride`` tokens after each chunk).

    Args:
        tokenizer: A ``TokenizerWrapper`` instance.
        window_size: Number of tokens per chunk (N).
        stride: Step between consecutive windows.  Defaults to ``window_size``
            (no overlap).  Use a smaller value for overlapping windows.
    """

    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        window_size: int = 512,
        stride: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size

        # Internal token buffer
        self._buffer: List[int] = []

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the internal token buffer."""
        self._buffer.clear()

    @property
    def buffer_size(self) -> int:
        """Current number of tokens sitting in the buffer."""
        return len(self._buffer)

    @property
    def buffer(self) -> List[int]:
        """Read-only view of the current token buffer."""
        return list(self._buffer)

    def chunk_texts(self, texts: List[str]) -> Iterator[List[int]]:
        """Tokenize *texts* and yield non-overlapping chunks on the fly.

        Yields:
            Lists of leq than ``window_size`` consecutive token IDs.
        """
        for text in texts:
            ids = self.tokenizer.encode(text)
            self._buffer.extend(ids)

            # Drain as many complete windows as possible from the buffer.
            while len(self._buffer) >= self.window_size:
                chunk = self._buffer[: self.window_size]
                self._buffer = self._buffer[self.stride :]
                yield chunk
            
            leftovers = self.flush()
            yield leftovers


    def flush(self) -> Optional[List[int]]:
        """Return the remaining buffer as a single (possibly smaller) chunk.

        Call this after the last call to :meth:`chunk_texts` to get the
        leftover tokens that didn't form a full window.

        Returns:
            The leftover token IDs, or ``None`` if the buffer is empty.
        """
        if not self._buffer:
            return None
        chunk = list(self._buffer)
        self._buffer.clear()
        return chunk

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over all chunks by calling :meth:`chunk_texts`."""
        return self

    def __next__(self) -> List[int]:
        """Not implemented — use :meth:`chunk_texts` or :meth:`iterate`."""
        raise NotImplementedError(
            "Use .chunk_texts(texts) or .iterate(texts) to consume chunks."
        )

    def iterate(
        self, texts: List[str]
    ) -> Iterator[List[int]]:
        """Convenience generator: chunk then flush leftovers.

        Yields:
            All full windows followed by the final leftover chunk (if any).
        """
        yield from self.chunk_texts(texts)
        leftover = self.flush()
        if leftover is not None:
            yield leftover

    # ── Representation ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"window_size={self.window_size}, "
            f"stride={self.stride}, "
            f"buffer_size={self.buffer_size})"
        )
    
if __name__ == "__main__":
    tokenizer = TokenizerWrapper()
    chunker = Chunking(tokenizer, 30, 1)
    i = 0
    for c in chunker.iterate(["Hello, i am from new orlean", "bruh idk what you said"]):
        i+=1
        
        print(tokenizer.decode(c, False), '====', c)
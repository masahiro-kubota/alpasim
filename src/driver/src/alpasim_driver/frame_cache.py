"""Session-specific helpers for async VAM driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from threading import RLock
from typing import List, Optional, cast

import numpy as np
import torch


@dataclass
class FrameEntry:
    """Represents a single camera frame and its cached tokenization."""

    timestamp_us: int
    image: Optional[np.ndarray]
    tokens: Optional[torch.Tensor]


def synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class FrameCache:
    """Keeps a bounded, time-ordered buffer of frames and tokens for a session.

    When subsample_factor > 1, the cache stores more frames than context_length
    to allow selecting every Nth frame for inference. This enables running
    inference at higher frequencies while maintaining the expected temporal
    spacing between frames that the model was trained on.

    Example with context_length=3, subsample_factor=2:
        - Buffer stores up to 3*2 = 6 frames
        - At inference, selects frames [newest, newest-2, newest-4]
        - Next inference selects [newest, newest-2, newest-4] from shifted buffer
    """

    context_length: int
    camera_id: str = ""
    subsample_factor: int = 1  # 1 = no subsampling, 2 = every other frame, etc.
    entries: List[FrameEntry] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @synchronized
    def add_image(self, timestamp_us: int, image: np.ndarray) -> None:
        """Insert or replace an image while keeping entries ordered by timestamp."""
        inserted = False
        # Iterate from newest to oldest since most inserts append.
        for offset, entry in enumerate(reversed(self.entries)):
            if entry.timestamp_us == timestamp_us:
                raise ValueError(f"Frame {timestamp_us} already exists in cache")
            if entry.timestamp_us < timestamp_us:
                insert_at = len(self.entries) - offset
                self.entries.insert(insert_at, FrameEntry(timestamp_us, image, None))
                inserted = True
                break
        if not inserted:
            self.entries.insert(0, FrameEntry(timestamp_us, image, None))

        self.prune()

    @synchronized
    def pending_frames(self) -> List[FrameEntry]:
        """Return frames that still need tokenization in timestamp order."""
        return [
            entry
            for entry in self.entries
            if entry.image is not None and entry.tokens is None
        ]

    @synchronized
    def frame_count(self) -> int:
        """Total number of frames currently cached."""
        return len(self.entries)

    def min_frames_required(self) -> int:
        """Minimum number of frames needed to construct a token window."""
        return (self.context_length - 1) * self.subsample_factor + 1

    @synchronized
    def has_enough_frames(self) -> bool:
        """Check if there are enough frames for a subsampled token window."""
        return len(self.entries) >= self.min_frames_required()

    @synchronized
    def token_count(self) -> int:
        """Number of frames that already have cached tokens."""
        return sum(1 for entry in self.entries if entry.tokens is not None)

    @synchronized
    def latest_token_window(self) -> List[torch.Tensor]:
        """Return the newest context window of tokens (oldest first).

        When subsample_factor > 1, selects every Nth frame starting from the
        newest frame and walking backwards. This maintains the expected temporal
        spacing between frames while allowing inference at higher frequencies.

        Returns:
            List of context_length token tensors, ordered oldest to newest.

        Raises:
            ValueError: If insufficient frames or tokens are available.
        """
        min_required = self.min_frames_required()
        assert len(self.entries) >= min_required, (
            f"Insufficient frames: have {len(self.entries)}, need at least "
            f"{min_required} (context_length={self.context_length}, "
            f"subsample_factor={self.subsample_factor})"
        )

        # Select frames: start from newest, walk backwards by subsample_factor
        selected_indices = []
        idx = len(self.entries) - 1  # Start at newest
        for _ in range(self.context_length):
            selected_indices.append(idx)
            idx -= self.subsample_factor

        # Reverse to get oldest-first order
        selected_indices = selected_indices[::-1]

        tokens = [self.entries[i].tokens for i in selected_indices]
        if any(token is None for token in tokens):
            raise ValueError(
                "Insufficient tokens: some selected frames are not yet tokenized"
            )

        return cast(List[torch.Tensor], tokens)

    @synchronized
    def prune(self) -> None:
        """Bound the cache to accommodate subsampled context queries."""
        max_entries = self.context_length * self.subsample_factor
        excess = len(self.entries) - max_entries
        if excess <= 0:
            return
        del self.entries[:excess]

    @synchronized
    def reset(self) -> None:
        """Clear all cached frames (e.g., when resetting the session)."""
        self.entries.clear()

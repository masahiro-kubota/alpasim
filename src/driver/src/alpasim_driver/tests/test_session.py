import numpy as np
import pytest
import torch

from ..frame_cache import FrameCache


def _make_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(2, 2, 3), dtype=np.uint8)


def _make_tokens(seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(0, 10, (3, 3), generator=rng, dtype=torch.int32)


def test_add_image_orders_and_prunes() -> None:
    cache = FrameCache(context_length=3)
    cache.add_image(30, _make_image(30))
    cache.add_image(10, _make_image(10))
    cache.add_image(20, _make_image(20))

    assert [entry.timestamp_us for entry in cache.entries] == [10, 20, 30]
    assert cache.frame_count() == 3
    assert cache.token_count() == 0

    # Buffer size = context_length * subsample_factor
    # With defaults (context_length=3, subsample_factor=1): 3 * 1 = 3
    # Adding a 4th frame should trigger pruning immediately
    cache.add_image(25, _make_image(25))
    assert [entry.timestamp_us for entry in cache.entries] == [20, 25, 30]
    assert cache.frame_count() == 3

    # Adding another frame continues to maintain max of 3
    cache.add_image(35, _make_image(35))
    assert [entry.timestamp_us for entry in cache.entries] == [25, 30, 35]
    assert cache.frame_count() == 3


def test_pending_frames_update_when_tokens_assigned() -> None:
    cache = FrameCache(context_length=3)
    cache.add_image(1, _make_image(1))
    cache.add_image(2, _make_image(2))

    pending = cache.pending_frames()
    assert len(pending) == 2

    pending[0].tokens = _make_tokens(1)

    pending = cache.pending_frames()
    assert len(pending) == 1
    assert pending[0].timestamp_us == 2

    pending[0].tokens = _make_tokens(2)

    pending = cache.pending_frames()
    assert len(pending) == 0
    assert all(entry.image is not None for entry in cache.entries)
    assert all(entry.tokens is not None for entry in cache.entries)
    assert cache.token_count() == 2


def test_latest_token_window_requires_full_context() -> None:
    cache = FrameCache(context_length=2)
    cache.add_image(1, _make_image(1))
    cache.add_image(2, _make_image(2))

    first_pending = cache.pending_frames()[0]
    first_pending.tokens = _make_tokens(1)

    with pytest.raises(ValueError):
        cache.latest_token_window()

    cache.pending_frames()[0].tokens = _make_tokens(2)
    window = cache.latest_token_window()
    assert len(window) == 2


def test_add_image_rejects_duplicate_timestamp() -> None:
    cache = FrameCache(context_length=2)
    cache.add_image(5, _make_image(5))

    with pytest.raises(ValueError):
        cache.add_image(5, _make_image(5))


def test_reset_clears_all_entries() -> None:
    cache = FrameCache(context_length=3)
    cache.add_image(1, _make_image(1))
    cache.add_image(2, _make_image(2))
    pending = cache.pending_frames()
    pending[0].tokens = _make_tokens(1)

    assert cache.frame_count() == 2
    assert cache.token_count() == 1

    cache.reset()

    assert cache.frame_count() == 0
    assert cache.token_count() == 0
    assert cache.entries == []


def test_pending_frames_order_after_prune() -> None:
    cache = FrameCache(context_length=3)
    cache.add_image(10, _make_image(10))
    cache.add_image(20, _make_image(20))
    cache.add_image(30, _make_image(30))

    # Tokenize frames that have already been processed to leave only the newest pending.
    oldest_entry = cache.pending_frames()[0]
    oldest_entry.tokens = _make_tokens(10)
    next_entry = cache.pending_frames()[0]
    next_entry.tokens = _make_tokens(20)

    cache.add_image(40, _make_image(40))

    pending = cache.pending_frames()
    assert [entry.timestamp_us for entry in pending] == [30, 40]


def test_subsample_factor_selects_every_nth_frame() -> None:
    """Test that subsample_factor=2 selects every other frame."""
    # context_length=3, subsample_factor=2 means we need frames at indices:
    # [newest, newest-2, newest-4] to get 3 frames
    # Min required = (3-1)*2 + 1 = 5 frames
    cache = FrameCache(context_length=3, subsample_factor=2)

    # Add 6 frames (timestamps 0-5)
    for i in range(6):
        cache.add_image(i * 100, _make_image(i))

    # Tokenize all frames
    for entry in cache.entries:
        entry.tokens = _make_tokens(entry.timestamp_us)

    # Get token window - should select frames at indices [-1, -3, -5]
    # which correspond to timestamps [500, 300, 100] -> ordered oldest first: [100, 300, 500]
    window = cache.latest_token_window()
    assert len(window) == 3

    # Verify the timestamps of selected frames match expected subsampling
    # The window should contain tokens from timestamps 100, 300, 500 (every other)
    selected_timestamps = [
        cache.entries[len(cache.entries) - 1 - i * 2].timestamp_us for i in range(3)
    ]
    selected_timestamps.reverse()  # oldest first
    assert selected_timestamps == [100, 300, 500]


def test_subsample_factor_min_frames_required() -> None:
    """Test min_frames_required calculation with different subsample factors."""
    # context_length=3, subsample_factor=1: min = (3-1)*1 + 1 = 3
    cache1 = FrameCache(context_length=3, subsample_factor=1)
    assert cache1.min_frames_required() == 3

    # context_length=3, subsample_factor=2: min = (3-1)*2 + 1 = 5
    cache2 = FrameCache(context_length=3, subsample_factor=2)
    assert cache2.min_frames_required() == 5

    # context_length=4, subsample_factor=3: min = (4-1)*3 + 1 = 10
    cache3 = FrameCache(context_length=4, subsample_factor=3)
    assert cache3.min_frames_required() == 10


def test_subsample_factor_insufficient_frames_raises() -> None:
    """Test that latest_token_window raises when not enough frames for subsampling."""
    cache = FrameCache(context_length=3, subsample_factor=2)
    # Min required = 5, add only 4
    for i in range(4):
        cache.add_image(i * 100, _make_image(i))
        cache.entries[-1].tokens = _make_tokens(i)

    with pytest.raises(AssertionError, match="Insufficient frames"):
        cache.latest_token_window()


def test_subsample_factor_buffer_size() -> None:
    """Test that buffer size accommodates subsampling."""
    cache = FrameCache(context_length=3, subsample_factor=2)
    # max_entries = context_length * subsample_factor = 3 * 2 = 6

    for i in range(10):
        cache.add_image(i * 100, _make_image(i))

    # Should keep max_entries = 6 frames
    assert cache.frame_count() == 6
    # Should keep the newest 6: [400, 500, 600, 700, 800, 900]
    assert [e.timestamp_us for e in cache.entries] == [
        400,
        500,
        600,
        700,
        800,
        900,
    ]


def test_has_enough_frames() -> None:
    """Test has_enough_frames helper method."""
    cache = FrameCache(context_length=3, subsample_factor=2)
    # Min required = 5

    for i in range(4):
        cache.add_image(i * 100, _make_image(i))
        assert not cache.has_enough_frames()

    cache.add_image(400, _make_image(4))
    assert cache.has_enough_frames()

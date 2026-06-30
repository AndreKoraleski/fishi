"""Memory-adaptive batching: shrink the chunk and retry on CUDA out-of-memory."""

from collections.abc import Callable

import structlog

logger = structlog.get_logger(__name__)


def predict_with_oom_backoff[T](
    items: list,
    predict_chunk: Callable[[list], list[T]],
    oom_error: type[BaseException],
    on_shrink: Callable[[], None] = lambda: None,
    start_batch: int | None = None,
) -> tuple[list[T], int]:
    """Run predict_chunk over items in chunks, halving the size on out-of-memory.

    Begins at start_batch (or the whole list). When predict_chunk raises oom_error it calls
    on_shrink (e.g. to empty the CUDA cache), halves the chunk size, and retries, re-raising only
    when a single item still will not fit.

    Parameters
    ----------
    items : list
        Inputs to run through predict_chunk, in order.
    predict_chunk : callable
        Maps a sublist of items to a list of results of the same length.
    oom_error : type of BaseException
        Exception that signals out-of-memory, e.g. torch.cuda.OutOfMemoryError.
    on_shrink : callable
        Called with no arguments before each retry, e.g. to free cached memory.
    start_batch : int, optional
        Initial chunk size. Defaults to the whole list.

    Returns
    -------
    results : list
        Per-item results aligned to items.
    limit : int
        Largest chunk size that fit, for the caller to reuse on the next call.
    """
    results: list = [None] * len(items)
    limit = start_batch or len(items) or 1
    index = 0
    while index < len(items):
        size = min(limit, len(items) - index)
        try:
            chunk = predict_chunk(items[index : index + size])
        except oom_error:
            on_shrink()
            if size == 1:
                raise
            limit = max(1, size // 2)
            logger.warning("oom_backoff", batch_size=limit)
            continue
        results[index : index + size] = chunk
        index += size
    return results, limit

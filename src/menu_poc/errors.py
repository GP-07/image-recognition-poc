"""Error kinds shared across components. A leaf module: it imports nothing from ``menu_poc``."""

from __future__ import annotations


class TransientError(RuntimeError):
    """A failure that may pass if the same work is tried again later (timeout, rate limit, 5xx).
    The pipeline retries the images that failed with one at the end of the run."""

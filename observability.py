"""Weave tracing bootstrap (mirrors the mise pattern).

An import-shim so `@op` works whether or not weave is installed/initialized:
- weave installed + init_weave() called  → real tracing
- weave installed, not initialized        → inert (a warning, no network)
- weave not installed                     → `op` is an identity decorator

So Phase 0 runs fine with no weave and no WANDB_API_KEY.
"""
import logging

log = logging.getLogger("retrieval_lab")

try:
    import weave as _weave

    op = _weave.op
except ImportError:  # weave not installed

    def op(fn=None, *_args, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn


def init_weave(project: str = "retrieval-lab") -> bool:
    """Turn tracing on. Returns True if live, False otherwise (never raises)."""
    try:
        import weave

        weave.init(project)
        log.info("Weave tracing on (project=%s)", project)
        return True
    except Exception as e:  # not installed, or no WANDB_API_KEY
        log.warning("Weave tracing off: %s", e)
        return False

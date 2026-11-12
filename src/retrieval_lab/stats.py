"""Interval estimates shared across phases. Deliberately light: no data, model
or cache import, so analysis code can use it without loading any of them.

One function for every proportion this repo reports, so "the intervals are
Wilson" is a property of the code rather than of whoever ran the numbers.
"""
from scipy.stats import binomtest


def wilson_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float] | None:
    """Wilson interval for `k` successes of `n`; None when `n == 0`.

    `method="wilson"` is passed by name because `proportion_ci` defaults to
    the exact (Clopper-Pearson) interval, which is wider. `binomtest(0, 0)`
    does not raise but returns nan, so zero rows is answered here with None.
    """
    if k < 0 or k > n:
        raise ValueError(f"wilson_ci: need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return None

    results = binomtest(k, n)
    ci = results.proportion_ci(
        confidence_level=level,
        method="wilson"
    )
    return (float(ci.low), float(ci.high))

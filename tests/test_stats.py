"""`wilson_ci` against known answers.

The interval is small, but it sits under every proportion this repo reports,
and each way it can go wrong is silent: the wrong method still returns a
plausible interval, and zero rows returns nan rather than raising.
"""
import pytest
from scipy.stats import binomtest

from retrieval_lab.stats import wilson_ci


def test_known_answer_16_of_20():
    low, high = wilson_ci(16, 20)
    assert low == pytest.approx(0.5840, abs=1e-4)
    assert high == pytest.approx(0.9193, abs=1e-4)


def test_it_is_wilson_not_the_exact_default():
    """`proportion_ci` defaults to the exact (Clopper-Pearson) interval, which
    is wider: (0.563, 0.943) at 16 of 20. Dropping `method="wilson"` would
    change every reported interval and raise nothing."""
    exact = binomtest(16, 20).proportion_ci()
    low, high = wilson_ci(16, 20)
    assert low > exact.low + 0.01
    assert high < exact.high - 0.01


def test_all_successes_reaches_exactly_one_and_stays_wide():
    """20 of 20 is not certainty: the simple (Wald) formula gives a zero-width
    interval here, Wilson gives about (0.839, 1.0)."""
    low, high = wilson_ci(20, 20)
    assert high == 1.0
    assert low == pytest.approx(0.8389, abs=1e-4)


def test_no_successes_reaches_exactly_zero():
    low, high = wilson_ci(0, 20)
    assert low == 0.0
    assert high == pytest.approx(1 - 0.8389, abs=1e-4)


def test_zero_rows_is_none_not_nan():
    """`binomtest(0, 0)` does not raise; its interval is (nan, nan). A judge
    that flags nothing has precision 0 of 0, so this is a real case, and a nan
    must not reach a table."""
    assert wilson_ci(0, 0) is None


def test_returns_plain_floats():
    """numpy scalars print as `np.float64(...)` wherever the tuple is shown."""
    low, high = wilson_ci(16, 20)
    assert type(low) is float and type(high) is float


def test_the_interval_is_lopsided_away_from_the_ceiling():
    """At 80% the interval reaches further down than up. A symmetric interval
    around 0.8 would be the Wald formula again."""
    low, high = wilson_ci(16, 20)
    assert (0.8 - low) > (high - 0.8)


def test_level_is_honoured():
    narrow, wide = wilson_ci(16, 20, level=0.80), wilson_ci(16, 20, level=0.99)
    assert wide[0] < narrow[0] and narrow[1] < wide[1]


def test_more_successes_than_rows_raises():
    with pytest.raises(ValueError):
        wilson_ci(21, 20)

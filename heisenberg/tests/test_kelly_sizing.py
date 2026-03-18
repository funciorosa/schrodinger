"""
HEISENBERG — Polymarket Arbitrage Bot
Tests for kelly_sizing.py

Run with:  pytest heisenberg/tests/test_kelly_sizing.py -v
"""

import math
import sys
import os

# Allow running from repo root or from the tests directory directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from kelly_sizing import KellyInput, KellyResult, KellySizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sizer() -> KellySizer:
    """Default KellySizer with standard parameters."""
    return KellySizer(kelly_fraction=0.25, max_position_pct=0.05, min_edge_threshold=0.01)


# ---------------------------------------------------------------------------
# 1. Full Kelly formula correctness for known inputs
# ---------------------------------------------------------------------------


def test_full_kelly_formula_correctness(sizer: KellySizer) -> None:
    """
    Verify f* = (p*b - q) / b for a canonical example.

    p=0.6, decimal odds=2.0  =>  b=1.0, q=0.4
    f* = (0.6*1.0 - 0.4) / 1.0 = 0.20
    """
    ki = KellyInput(
        prob_win=0.6,
        odds_win=2.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)

    assert math.isclose(result.full_kelly, 0.20, rel_tol=1e-9), (
        f"Expected full_kelly=0.20, got {result.full_kelly}"
    )


def test_full_kelly_formula_second_example(sizer: KellySizer) -> None:
    """
    Second known input: p=0.55, decimal odds=1.9
    b = 0.9, q = 0.45
    f* = (0.55*0.9 - 0.45) / 0.9 = (0.495 - 0.45) / 0.9 = 0.05
    """
    ki = KellyInput(
        prob_win=0.55,
        odds_win=1.9,
        odds_lose=1.0,
        bankroll=5_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)

    expected_fk = (0.55 * 0.9 - 0.45) / 0.9
    assert math.isclose(result.full_kelly, expected_fk, rel_tol=1e-9), (
        f"Expected full_kelly={expected_fk:.6f}, got {result.full_kelly}"
    )


# ---------------------------------------------------------------------------
# 2. Fractional Kelly is exactly 0.25x full Kelly
# ---------------------------------------------------------------------------


def test_fractional_kelly_is_quarter_of_full(sizer: KellySizer) -> None:
    """fractional_kelly must equal full_kelly * 0.25 exactly."""
    ki = KellyInput(
        prob_win=0.65,
        odds_win=2.5,
        odds_lose=1.0,
        bankroll=20_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)

    assert math.isclose(result.fractional_kelly, result.full_kelly * 0.25, rel_tol=1e-12), (
        f"fractional_kelly {result.fractional_kelly} != full_kelly * 0.25 {result.full_kelly * 0.25}"
    )


def test_custom_kelly_fraction_applied(sizer: KellySizer) -> None:
    """If KellyInput overrides kelly_fraction, the override is honoured."""
    ki = KellyInput(
        prob_win=0.60,
        odds_win=2.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.50,  # half Kelly
    )
    result = sizer.compute_kelly(ki)

    assert math.isclose(result.fractional_kelly, result.full_kelly * 0.50, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 3. Negative edge returns 0 position size
# ---------------------------------------------------------------------------


def test_negative_edge_returns_zero_position(sizer: KellySizer) -> None:
    """
    When the implied edge is negative, position_size must be 0.

    p=0.40, decimal odds=2.0  =>  b=1.0, q=0.60
    f* = (0.40 - 0.60) / 1.0 = -0.20  (negative edge)
    """
    ki = KellyInput(
        prob_win=0.40,
        odds_win=2.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)

    assert result.position_size == 0.0, (
        f"Expected position_size=0.0 for negative edge, got {result.position_size}"
    )
    assert result.fractional_kelly == 0.0
    assert result.max_loss == 0.0
    assert result.risk_pct == 0.0


def test_breakeven_edge_returns_zero_position(sizer: KellySizer) -> None:
    """
    Exactly breakeven (full_kelly == 0) must also return 0 position.

    p=0.50, decimal odds=2.0  =>  b=1.0, q=0.50
    f* = (0.50 - 0.50) / 1.0 = 0.0
    """
    ki = KellyInput(
        prob_win=0.50,
        odds_win=2.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)

    assert result.position_size == 0.0


# ---------------------------------------------------------------------------
# 4. Position size capped at max_position_pct
# ---------------------------------------------------------------------------


def test_position_size_capped_at_max(sizer: KellySizer) -> None:
    """
    A very high-edge bet should be capped at max_position_pct * bankroll.

    p=0.99, decimal odds=100.0 gives an astronomically high Kelly fraction.
    With max_position_pct=0.05 and bankroll=10_000, cap = 500.
    """
    ki = KellyInput(
        prob_win=0.99,
        odds_win=100.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)

    max_allowed = sizer.max_position_pct * ki.bankroll  # 500.0
    assert result.position_size <= max_allowed + 1e-9, (
        f"position_size {result.position_size} exceeds cap {max_allowed}"
    )
    assert math.isclose(result.position_size, max_allowed, rel_tol=1e-9)


def test_position_size_not_capped_when_small(sizer: KellySizer) -> None:
    """
    A modest edge should NOT be capped — raw fractional position should come through.

    p=0.52, decimal odds=2.0  =>  b=1.0, q=0.48
    f* = 0.04, frac=0.01, position = 0.01 * 10_000 = 100 < 500 cap.
    """
    ki = KellyInput(
        prob_win=0.52,
        odds_win=2.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)

    expected_fk = (0.52 * 1.0 - 0.48) / 1.0        # 0.04
    expected_frac = expected_fk * 0.25              # 0.01
    expected_pos = expected_frac * 10_000.0         # 100.0
    max_allowed = sizer.max_position_pct * ki.bankroll  # 500.0

    assert expected_pos < max_allowed, "Precondition: this test expects an uncapped position."
    assert math.isclose(result.position_size, expected_pos, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 5. Continuous Kelly from EV / variance
# ---------------------------------------------------------------------------


def test_continuous_kelly_from_ev_variance(sizer: KellySizer) -> None:
    """
    f* = EV / variance.  With EV=0.10 and variance=0.50:
    full_kelly = 0.20, fractional = 0.05, position = 0.05 * 10_000 = 500.
    Cap is also 500, so position == 500.
    """
    result = sizer.compute_kelly_from_ev(ev=0.10, variance=0.50, bankroll=10_000.0)

    assert math.isclose(result.full_kelly, 0.20, rel_tol=1e-9)
    assert math.isclose(result.fractional_kelly, 0.05, rel_tol=1e-9)


def test_continuous_kelly_negative_ev_returns_zero(sizer: KellySizer) -> None:
    """Negative EV must yield zero position."""
    result = sizer.compute_kelly_from_ev(ev=-0.05, variance=0.30, bankroll=10_000.0)

    assert result.position_size == 0.0
    assert result.full_kelly < 0.0


def test_continuous_kelly_zero_variance_returns_zero(sizer: KellySizer) -> None:
    """Zero variance is undefined; should return zero position safely."""
    result = sizer.compute_kelly_from_ev(ev=0.10, variance=0.0, bankroll=10_000.0)

    assert result.position_size == 0.0
    assert result.full_kelly == 0.0


# ---------------------------------------------------------------------------
# 6. Portfolio Kelly sums to <= 20% of bankroll
# ---------------------------------------------------------------------------


def test_portfolio_kelly_sum_within_cap(sizer: KellySizer) -> None:
    """
    Total dollar exposure across all positions must be <= 20% of bankroll.
    """
    signals = [
        (0.60, 2.0),
        (0.55, 1.8),
        (0.65, 3.0),
        (0.70, 2.2),
    ]
    bankroll = 50_000.0
    positions = sizer.compute_portfolio_kelly(signals, bankroll)

    total = sum(positions)
    assert total <= 0.20 * bankroll + 1e-6, (
        f"Portfolio total {total} exceeds 20% cap {0.20 * bankroll}"
    )


def test_portfolio_kelly_returns_one_per_signal(sizer: KellySizer) -> None:
    """Output list length must equal input signals length."""
    signals = [(0.60, 2.0), (0.55, 1.8), (0.40, 2.0)]  # last one is negative edge
    positions = sizer.compute_portfolio_kelly(signals, bankroll=10_000.0)

    assert len(positions) == len(signals)


def test_portfolio_kelly_negative_edge_gets_zero(sizer: KellySizer) -> None:
    """Signals with negative edge should receive 0 allocation."""
    signals = [
        (0.60, 2.0),   # positive edge
        (0.30, 1.5),   # negative edge: b=0.5, f* = (0.3*0.5-0.7)/0.5 < 0
    ]
    positions = sizer.compute_portfolio_kelly(signals, bankroll=10_000.0)

    assert positions[1] == 0.0, (
        f"Negative-edge signal received non-zero allocation: {positions[1]}"
    )


def test_portfolio_kelly_all_negative_returns_zeros(sizer: KellySizer) -> None:
    """All-negative-edge portfolio must return all zeros."""
    signals = [(0.30, 1.5), (0.35, 1.4)]
    positions = sizer.compute_portfolio_kelly(signals, bankroll=10_000.0)

    assert all(p == 0.0 for p in positions)


# ---------------------------------------------------------------------------
# 7. Invalid inputs handled gracefully
# ---------------------------------------------------------------------------


def test_validate_inputs_prob_zero(sizer: KellySizer) -> None:
    """prob=0 is invalid — validate_inputs must return False."""
    assert sizer.validate_inputs(prob=0.0, odds=2.0) is False


def test_validate_inputs_prob_one(sizer: KellySizer) -> None:
    """prob=1 is invalid — validate_inputs must return False."""
    assert sizer.validate_inputs(prob=1.0, odds=2.0) is False


def test_validate_inputs_odds_equal_one(sizer: KellySizer) -> None:
    """odds=1.0 (no profit) is invalid — validate_inputs must return False."""
    assert sizer.validate_inputs(prob=0.55, odds=1.0) is False


def test_validate_inputs_odds_below_one(sizer: KellySizer) -> None:
    """odds < 1.0 is invalid — validate_inputs must return False."""
    assert sizer.validate_inputs(prob=0.55, odds=0.5) is False


def test_compute_kelly_invalid_prob_zero(sizer: KellySizer) -> None:
    """compute_kelly with prob=0 must return zero-position result without raising."""
    ki = KellyInput(
        prob_win=0.0,
        odds_win=2.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)
    assert result.position_size == 0.0


def test_compute_kelly_invalid_prob_one(sizer: KellySizer) -> None:
    """compute_kelly with prob=1 must return zero-position result without raising."""
    ki = KellyInput(
        prob_win=1.0,
        odds_win=2.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)
    assert result.position_size == 0.0


def test_compute_kelly_invalid_odds_below_one(sizer: KellySizer) -> None:
    """compute_kelly with odds < 1 must return zero-position result without raising."""
    ki = KellyInput(
        prob_win=0.60,
        odds_win=0.8,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)
    assert result.position_size == 0.0


# ---------------------------------------------------------------------------
# 8. risk_pct and max_loss consistency
# ---------------------------------------------------------------------------


def test_risk_pct_consistent_with_position_and_bankroll(sizer: KellySizer) -> None:
    """risk_pct must equal position_size / bankroll."""
    ki = KellyInput(
        prob_win=0.60,
        odds_win=2.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)

    expected_risk_pct = result.position_size / ki.bankroll
    assert math.isclose(result.risk_pct, expected_risk_pct, rel_tol=1e-12)


def test_max_loss_equals_position_size(sizer: KellySizer) -> None:
    """max_loss must equal position_size (worst case: full stake lost)."""
    ki = KellyInput(
        prob_win=0.60,
        odds_win=2.0,
        odds_lose=1.0,
        bankroll=10_000.0,
        kelly_fraction=0.25,
    )
    result = sizer.compute_kelly(ki)

    assert math.isclose(result.max_loss, result.position_size, rel_tol=1e-12)

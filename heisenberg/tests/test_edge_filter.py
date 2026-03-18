"""
test_edge_filter.py — HEISENBERG Project
EDGE_AGENT: Pytest suite for edge_filter.py
"""

from __future__ import annotations

import math
import sys
import os

import pytest

# Allow importing from the parent heisenberg package directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from edge_filter import EdgeFilter, EdgeSignal, SpreadData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ef() -> EdgeFilter:
    """Default EdgeFilter with standard thresholds."""
    return EdgeFilter(min_edge_bps=50, min_z_score=1.5, max_spread_bps=500)


# ---------------------------------------------------------------------------
# Z-score tests
# ---------------------------------------------------------------------------

def test_z_score_constant_history_returns_zero(ef: EdgeFilter) -> None:
    """Z-score must be 0.0 when all prices in history are identical (std == 0)."""
    history = [0.50] * 20
    result = ef.compute_z_score(0.50, history)
    assert result == 0.0


def test_z_score_correct_for_known_mean_std(ef: EdgeFilter) -> None:
    """
    Z-score should equal (price - mean) / std for a known distribution.
    Use history [1, 2, 3, 4, 5]: mean=3, std(ddof=1)=sqrt(2.5).
    Price=5 → z = (5-3)/sqrt(2.5) ≈ 1.2649
    """
    history = [1.0, 2.0, 3.0, 4.0, 5.0]
    price = 5.0
    mean = 3.0
    std = math.sqrt(sum((x - mean) ** 2 for x in history) / (len(history) - 1))
    expected_z = (price - mean) / std
    result = ef.compute_z_score(price, history, window=len(history))
    assert math.isclose(result, expected_z, rel_tol=1e-9)


def test_z_score_window_smaller_than_two_returns_zero(ef: EdgeFilter) -> None:
    """Window of 1 has no meaningful std; must return 0.0."""
    history = [0.55, 0.60, 0.65]
    result = ef.compute_z_score(0.70, history, window=1)
    assert result == 0.0


def test_z_score_empty_history_returns_zero(ef: EdgeFilter) -> None:
    """Empty price history must return 0.0 without raising."""
    result = ef.compute_z_score(0.50, [])
    assert result == 0.0


def test_z_score_uses_only_window_tail(ef: EdgeFilter) -> None:
    """
    Only the last `window` entries should be used.
    Prepend outliers that must be excluded by the window.
    """
    outliers = [1000.0] * 100
    recent = [0.50, 0.50, 0.50, 0.50, 0.50]
    history = outliers + recent
    # With window=5 the outliers are excluded → std=0 → z=0
    result = ef.compute_z_score(0.50, history, window=5)
    assert result == 0.0


# ---------------------------------------------------------------------------
# EV tests
# ---------------------------------------------------------------------------

def test_ev_negative_when_fees_dominate(ef: EdgeFilter) -> None:
    """
    When the market is fairly priced (prob == market probability) and fees are
    high, EV should be negative.

    prob=0.5, ask=0.50 → odds_yes=2.0, bid=0.50 → odds_no=1/(0.5)=2.0
    EV = 0.5*2 - 0.5*2 - fee = -fee < 0
    """
    prob = 0.5
    ask = 0.50
    bid = 0.50
    odds_yes = 1.0 / ask
    odds_no = 1.0 / (1.0 - bid)
    ev = ef.compute_ev(prob, odds_yes, odds_no, fee_bps=200)
    assert ev < 0.0


def test_ev_positive_for_favorable_odds(ef: EdgeFilter) -> None:
    """
    When the model assigns higher probability than the market implies and
    fees are low, EV should be positive.

    prob=0.80, ask=0.60 → odds_yes≈1.667, bid=0.70 → odds_no≈3.333
    EV = 0.80*1.667 - 0.20*3.333 - 0.002 = 1.333 - 0.667 - 0.002 = 0.665
    """
    prob = 0.80
    ask = 0.60
    bid = 0.70
    odds_yes = 1.0 / ask
    odds_no = 1.0 / (1.0 - bid)
    ev = ef.compute_ev(prob, odds_yes, odds_no, fee_bps=20)
    assert ev > 0.0


def test_ev_formula_exact(ef: EdgeFilter) -> None:
    """Verify the EV formula against a manually computed value."""
    prob = 0.70
    odds_yes = 2.0   # ask = 0.50
    odds_no = 4.0    # bid = 0.75
    fee_bps = 30
    expected = 0.70 * 2.0 - 0.30 * 4.0 - 30 / 10_000
    result = ef.compute_ev(prob, odds_yes, odds_no, fee_bps=fee_bps)
    assert math.isclose(result, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Net edge tests
# ---------------------------------------------------------------------------

def test_net_edge_combines_components_correctly(ef: EdgeFilter) -> None:
    """
    Manually verify the net_edge formula with known inputs.

    z_score=3.0  → clamp(3/3,-1,1)=1.0  → 0.4*1.0 = 0.40
    ev=0.10      → clamp(0.10*10,-1,1)=1.0 → 0.4*1.0 = 0.40
    posterior=1.0 → 0.2*(2*1-1)=0.2*1=0.20
    net_edge = 0.40 + 0.40 + 0.20 = 1.00
    """
    result = ef.compute_net_edge(z_score=3.0, ev=0.10, bayesian_posterior=1.0)
    assert math.isclose(result, 1.00, rel_tol=1e-9)


def test_net_edge_neutral_posterior(ef: EdgeFilter) -> None:
    """Posterior=0.5 contributes 0 to net edge (perfectly neutral)."""
    # posterior=0.5 → 0.2*(2*0.5-1)=0.0
    result_with = ef.compute_net_edge(z_score=1.5, ev=0.05, bayesian_posterior=0.5)
    result_without_bayes = 0.4 * (1.5 / 3.0) + 0.4 * min(0.05 * 10, 1.0)
    assert math.isclose(result_with, result_without_bayes, rel_tol=1e-9)


def test_net_edge_clamping_limits_extremes(ef: EdgeFilter) -> None:
    """Very large z_score and ev must be clamped so net_edge stays <= 1.0."""
    result = ef.compute_net_edge(z_score=1000.0, ev=1000.0, bayesian_posterior=1.0)
    # max possible: 0.4*1 + 0.4*1 + 0.2*1 = 1.0
    assert math.isclose(result, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# filter() tests
# ---------------------------------------------------------------------------

def test_filter_blocks_insufficient_z_score(ef: EdgeFilter) -> None:
    """Trade must be blocked when abs(z_score) < min_z_score."""
    spread = ef.compute_spread(bid=0.48, ask=0.52)  # 400 bps — within limit
    signal: EdgeSignal = ef.filter(spread=spread, z_score=0.5, ev=0.10, posterior=0.9)
    assert signal.is_tradeable is False


def test_filter_blocks_wide_spread(ef: EdgeFilter) -> None:
    """Trade must be blocked when spread_bps > max_spread_bps."""
    spread = ef.compute_spread(bid=0.30, ask=0.80)  # 5000 bps — exceeds 500 bps limit
    signal: EdgeSignal = ef.filter(spread=spread, z_score=3.0, ev=0.10, posterior=0.9)
    assert signal.is_tradeable is False


def test_filter_blocks_insufficient_net_edge(ef: EdgeFilter) -> None:
    """Trade must be blocked when net_edge <= min_edge_bps/10000."""
    # Drive net_edge negative: z_score negative, low ev, low posterior
    spread = ef.compute_spread(bid=0.48, ask=0.52)  # 400 bps
    signal: EdgeSignal = ef.filter(spread=spread, z_score=-3.0, ev=-0.10, posterior=0.0)
    assert signal.is_tradeable is False


def test_filter_allows_valid_trade(ef: EdgeFilter) -> None:
    """Trade must be allowed when all conditions are met."""
    spread = ef.compute_spread(bid=0.48, ask=0.52)  # 400 bps — within limit
    # z=3.0, ev=0.10, posterior=1.0 → net_edge=1.0 >> 0.0050
    signal: EdgeSignal = ef.filter(spread=spread, z_score=3.0, ev=0.10, posterior=1.0)
    assert signal.is_tradeable is True


def test_filter_signal_fields_populated(ef: EdgeFilter) -> None:
    """EdgeSignal must have all fields set with reasonable types."""
    spread = ef.compute_spread(bid=0.48, ask=0.52)
    signal: EdgeSignal = ef.filter(spread=spread, z_score=2.0, ev=0.05, posterior=0.8)
    assert isinstance(signal.z_score, float)
    assert isinstance(signal.expected_value, float)
    assert isinstance(signal.net_edge, float)
    assert isinstance(signal.is_tradeable, bool)
    assert 0.0 <= signal.confidence <= 1.0


# ---------------------------------------------------------------------------
# compute_spread tests
# ---------------------------------------------------------------------------

def test_compute_spread_values(ef: EdgeFilter) -> None:
    """Verify mid, spread, and spread_bps are computed correctly."""
    spread = ef.compute_spread(bid=0.48, ask=0.52)
    assert math.isclose(spread.mid, 0.50, rel_tol=1e-9)
    assert math.isclose(spread.spread, 0.04, rel_tol=1e-9)
    assert spread.spread_bps == 400


def test_compute_spread_zero_spread(ef: EdgeFilter) -> None:
    """Bid == ask should produce zero spread."""
    spread = ef.compute_spread(bid=0.50, ask=0.50)
    assert spread.spread == 0.0
    assert spread.spread_bps == 0


# ---------------------------------------------------------------------------
# Edge-case / regression tests
# ---------------------------------------------------------------------------

def test_handles_single_element_history(ef: EdgeFilter) -> None:
    """Single-element history cannot form a std — must return 0.0."""
    result = ef.compute_z_score(0.55, [0.50])
    assert result == 0.0


def test_z_score_negative_for_below_mean_price(ef: EdgeFilter) -> None:
    """Price below the historical mean must yield a negative z-score."""
    history = [0.60, 0.62, 0.64, 0.66, 0.68]
    result = ef.compute_z_score(0.50, history)
    assert result < 0.0


def test_filter_confidence_is_abs_net_edge_clamped(ef: EdgeFilter) -> None:
    """Confidence must equal abs(net_edge) clamped to [0, 1]."""
    spread = ef.compute_spread(bid=0.48, ask=0.52)
    signal: EdgeSignal = ef.filter(spread=spread, z_score=1.0, ev=0.03, posterior=0.6)
    expected_confidence = min(abs(signal.net_edge), 1.0)
    assert math.isclose(signal.confidence, expected_confidence, rel_tol=1e-9)

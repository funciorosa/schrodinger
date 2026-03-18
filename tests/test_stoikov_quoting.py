"""
test_stoikov_quoting.py — pytest suite for stoikov_quoting.py
HEISENBERG / Polymarket arbitrage bot
"""

from __future__ import annotations

import math
import sys
import os

# Allow imports from the heisenberg package when running pytest from the
# repository root or from the tests directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from stoikov_quoting import QuoteResult, StoikovParams, StoikovQuoter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params() -> StoikovParams:
    """Standard parameters used across most tests."""
    return StoikovParams(gamma=0.1, sigma=0.02, T=300.0, dt=1.0, kappa=1.5)


@pytest.fixture
def quoter(default_params: StoikovParams) -> StoikovQuoter:
    """StoikovQuoter with default parameters."""
    return StoikovQuoter(default_params)


# ---------------------------------------------------------------------------
# 1. Reservation price decreases with positive inventory (long skew)
# ---------------------------------------------------------------------------

def test_reservation_price_decreases_with_positive_inventory(quoter: StoikovQuoter) -> None:
    """A net-long position should push the reservation price below the mid."""
    mid = 0.50
    t = 0.0

    r_zero = quoter.compute_reservation_price(mid, inventory=0.0, t=t)
    r_long = quoter.compute_reservation_price(mid, inventory=0.5, t=t)
    r_max_long = quoter.compute_reservation_price(mid, inventory=1.0, t=t)

    assert r_long < r_zero, (
        "Positive inventory should reduce reservation price below mid"
    )
    assert r_max_long < r_long, (
        "Larger positive inventory should reduce reservation price further"
    )


# ---------------------------------------------------------------------------
# 2. Reservation price equals mid at zero inventory
# ---------------------------------------------------------------------------

def test_reservation_price_equals_mid_at_zero_inventory(quoter: StoikovQuoter) -> None:
    """With zero inventory there is no skew correction; r must equal mid."""
    for mid in [0.10, 0.30, 0.50, 0.70, 0.90]:
        r = quoter.compute_reservation_price(mid, inventory=0.0, t=0.0)
        assert math.isclose(r, mid, abs_tol=1e-9), (
            f"Expected r == mid={mid} at zero inventory, got {r}"
        )


# ---------------------------------------------------------------------------
# 3. Optimal spread increases with higher risk aversion (gamma)
# ---------------------------------------------------------------------------

def test_optimal_spread_increases_with_gamma() -> None:
    """Higher gamma should produce a wider optimal spread."""
    t = 0.0
    spreads = []
    for gamma in [0.05, 0.10, 0.20, 0.50]:
        params = StoikovParams(gamma=gamma, sigma=0.02, T=300.0, kappa=1.5)
        q = StoikovQuoter(params)
        spreads.append(q.compute_optimal_spread(t))

    for i in range(len(spreads) - 1):
        assert spreads[i] < spreads[i + 1], (
            f"Spread should be wider for larger gamma: "
            f"spreads[{i}]={spreads[i]:.6f} >= spreads[{i+1}]={spreads[i+1]:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. Spread decreases as time approaches T (approaching expiry)
# ---------------------------------------------------------------------------

def test_spread_decreases_as_time_approaches_T(quoter: StoikovQuoter) -> None:
    """The inventory-risk term shrinks as (T - t) → 0, so spread should narrow."""
    # The arrival-rate term is constant; the inventory-risk term drives the
    # decrease.  We compare spread at t=0 vs t close to T.
    spread_early = quoter.compute_optimal_spread(t=0.0)
    spread_late = quoter.compute_optimal_spread(t=290.0)   # 10 s remaining
    spread_end = quoter.compute_optimal_spread(t=300.0)    # 0 s remaining

    assert spread_early > spread_late, (
        "Spread at t=0 should be wider than at t=290"
    )
    assert spread_late >= spread_end, (
        "Spread at t=290 should be >= spread at t=T"
    )


# ---------------------------------------------------------------------------
# 5. Quotes stay within [0.01, 0.99]
# ---------------------------------------------------------------------------

def test_quotes_stay_within_binary_bounds(quoter: StoikovQuoter) -> None:
    """Bid and ask must always lie in [0.01, 0.99]."""
    test_cases = [
        (0.01, -1.0, 0.0),    # extreme low mid, max short inventory
        (0.99,  1.0, 0.0),    # extreme high mid, max long inventory
        (0.50,  0.0, 150.0),  # mid-point, no inventory, mid-time
        (0.50,  1.0, 299.0),  # near expiry, max long
        (0.50, -1.0, 299.0),  # near expiry, max short
    ]
    for mid, inv, t in test_cases:
        result = quoter.compute_quotes(mid, inv, t)
        assert result.bid_quote >= 0.01, (
            f"bid={result.bid_quote} < 0.01 for mid={mid}, inv={inv}, t={t}"
        )
        assert result.ask_quote <= 0.99, (
            f"ask={result.ask_quote} > 0.99 for mid={mid}, inv={inv}, t={t}"
        )
        assert result.bid_quote <= 0.99
        assert result.ask_quote >= 0.01


# ---------------------------------------------------------------------------
# 6. Bid is always strictly less than ask
# ---------------------------------------------------------------------------

def test_bid_always_less_than_ask(quoter: StoikovQuoter) -> None:
    """The bid quote must be strictly below the ask quote."""
    test_cases = [
        (0.50,  0.0,   0.0),
        (0.50,  0.5,  60.0),
        (0.50, -0.5,  60.0),
        (0.10,  0.0, 150.0),
        (0.90,  0.0, 150.0),
        (0.50,  1.0, 299.9),
        (0.50, -1.0, 299.9),
    ]
    for mid, inv, t in test_cases:
        result = quoter.compute_quotes(mid, inv, t)
        assert result.bid_quote < result.ask_quote, (
            f"bid={result.bid_quote} >= ask={result.ask_quote} "
            f"for mid={mid}, inv={inv}, t={t}"
        )


# ---------------------------------------------------------------------------
# 7. Inventory skew is non-negative
# ---------------------------------------------------------------------------

def test_inventory_skew_is_non_negative(quoter: StoikovQuoter) -> None:
    """inventory_skew = abs(inventory) must never be negative."""
    for inventory in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        result = quoter.compute_quotes(0.50, inventory, t=0.0)
        assert result.inventory_skew >= 0.0, (
            f"inventory_skew={result.inventory_skew} < 0 for inventory={inventory}"
        )


def test_inventory_skew_is_abs_of_inventory(quoter: StoikovQuoter) -> None:
    """inventory_skew should equal abs(inventory) exactly."""
    for inventory in [-0.75, -0.25, 0.0, 0.25, 0.75]:
        result = quoter.compute_quotes(0.50, inventory, t=0.0)
        assert math.isclose(result.inventory_skew, abs(inventory), abs_tol=1e-12), (
            f"Expected skew={abs(inventory)}, got {result.inventory_skew}"
        )


# ---------------------------------------------------------------------------
# 8. Binary market adjustment enforces minimum spread of 0.01
# ---------------------------------------------------------------------------

def test_binary_market_adjustment_enforces_minimum_spread(quoter: StoikovQuoter) -> None:
    """adjust_for_binary_market must ensure spread >= 0.01."""
    # Construct a QuoteResult that would have a sub-minimum spread
    tiny_spread_quotes = QuoteResult(
        reservation_price=0.50,
        bid_quote=0.495,
        ask_quote=0.501,   # spread = 0.006 < 0.01
        spread=0.006,
        inventory_skew=0.0,
    )
    adjusted = quoter.adjust_for_binary_market(tiny_spread_quotes)
    assert adjusted.spread >= 0.01, (
        f"Adjusted spread {adjusted.spread} is below minimum 0.01"
    )
    assert adjusted.bid_quote < adjusted.ask_quote


def test_binary_market_adjustment_on_degenerate_equal_quotes(quoter: StoikovQuoter) -> None:
    """adjust_for_binary_market handles bid == ask (zero spread)."""
    zero_spread = QuoteResult(
        reservation_price=0.50,
        bid_quote=0.50,
        ask_quote=0.50,
        spread=0.0,
        inventory_skew=0.0,
    )
    adjusted = quoter.adjust_for_binary_market(zero_spread)
    assert adjusted.spread >= 0.01
    assert adjusted.bid_quote < adjusted.ask_quote
    assert adjusted.bid_quote >= StoikovQuoter.MIN_PRICE
    assert adjusted.ask_quote <= StoikovQuoter.MAX_PRICE


# ---------------------------------------------------------------------------
# 9. Symmetric inventory produces symmetric quote adjustment
# ---------------------------------------------------------------------------

def test_symmetric_inventory_produces_symmetric_skew(quoter: StoikovQuoter) -> None:
    """Positive and negative inventory of equal magnitude should have the same
    inventory_skew and produce quotes that are reflections around the mid."""
    mid = 0.50
    t = 30.0

    result_long = quoter.compute_quotes(mid, inventory=0.3, t=t)
    result_short = quoter.compute_quotes(mid, inventory=-0.3, t=t)

    assert math.isclose(result_long.inventory_skew, result_short.inventory_skew, abs_tol=1e-12)

    # Reservation prices should be symmetric around mid
    deviation_long = mid - result_long.reservation_price
    deviation_short = result_short.reservation_price - mid
    assert math.isclose(deviation_long, deviation_short, abs_tol=1e-9), (
        f"Reservation prices are not symmetric: "
        f"long deviation={deviation_long}, short deviation={deviation_short}"
    )


# ---------------------------------------------------------------------------
# 10. Spread field in QuoteResult matches ask - bid
# ---------------------------------------------------------------------------

def test_spread_field_consistent_with_bid_ask(quoter: StoikovQuoter) -> None:
    """The spread field in QuoteResult must equal ask_quote - bid_quote."""
    test_cases = [
        (0.50,  0.0,   0.0),
        (0.30,  0.2,  45.0),
        (0.70, -0.4, 200.0),
    ]
    for mid, inv, t in test_cases:
        result = quoter.compute_quotes(mid, inv, t)
        expected_spread = result.ask_quote - result.bid_quote
        assert math.isclose(result.spread, expected_spread, abs_tol=1e-12), (
            f"spread field {result.spread} != ask-bid {expected_spread}"
        )

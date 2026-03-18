"""
HEISENBERG — Polymarket Arbitrage Bot
Kelly Criterion Position Sizing Module

Implements fractional Kelly criterion for optimal bet sizing,
continuous Kelly from EV/variance, and multi-asset portfolio Kelly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class KellyInput:
    prob_win: float
    odds_win: float
    odds_lose: float
    bankroll: float
    kelly_fraction: float = 0.25


@dataclass
class KellyResult:
    full_kelly: float
    fractional_kelly: float
    position_size: float
    max_loss: float
    risk_pct: float


class KellySizer:
    """
    Computes Kelly criterion position sizes for Polymarket arbitrage signals.

    Parameters
    ----------
    kelly_fraction : float
        Fractional Kelly multiplier (default 0.25 = quarter Kelly).
    max_position_pct : float
        Hard cap on any single position as a fraction of bankroll (default 5%).
    min_edge_threshold : float
        Minimum edge required before sizing a position (default 1%).
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.05,
        min_edge_threshold: float = 0.01,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_edge_threshold = min_edge_threshold

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_inputs(self, prob: float, odds: float) -> bool:
        """
        Return True only when inputs are within acceptable bounds.

        Parameters
        ----------
        prob : float
            Win probability — must be strictly inside (0, 1).
        odds : float
            Decimal odds for a win — must be strictly greater than 1.
        """
        if not (0.0 < prob < 1.0):
            return False
        if odds <= 1.0:
            return False
        return True

    # ------------------------------------------------------------------
    # Standard Kelly (discrete bets)
    # ------------------------------------------------------------------

    def compute_kelly(self, input: KellyInput) -> KellyResult:
        """
        Compute Kelly position size for a binary bet.

        Formula: f* = (p * b - q) / b
            where b = net decimal odds (odds_win - 1),
                  p = prob_win,
                  q = 1 - p

        Parameters
        ----------
        input : KellyInput
            All parameters required for Kelly computation.

        Returns
        -------
        KellyResult
            Sizing result with full_kelly, fractional_kelly, position_size,
            max_loss, and risk_pct fields populated.
        """
        p = input.prob_win
        q = 1.0 - p
        # b is the NET odds (profit per unit staked), i.e. decimal_odds - 1
        b = input.odds_win - 1.0

        if b <= 0.0 or not self.validate_inputs(p, input.odds_win):
            return KellyResult(
                full_kelly=0.0,
                fractional_kelly=0.0,
                position_size=0.0,
                max_loss=0.0,
                risk_pct=0.0,
            )

        full_kelly = (p * b - q) / b

        # Negative edge — no position
        if full_kelly <= 0.0:
            return KellyResult(
                full_kelly=full_kelly,
                fractional_kelly=0.0,
                position_size=0.0,
                max_loss=0.0,
                risk_pct=0.0,
            )

        # Use the fraction provided in input (allows per-call override)
        frac = input.kelly_fraction
        fractional_kelly = full_kelly * frac

        # Raw position size
        raw_position = fractional_kelly * input.bankroll

        # Clamp to max_position_pct * bankroll
        max_allowed = self.max_position_pct * input.bankroll
        position_size = min(raw_position, max_allowed)

        # max_loss is the full position staked (worst case: lose it all)
        max_loss = position_size

        risk_pct = position_size / input.bankroll if input.bankroll > 0.0 else 0.0

        return KellyResult(
            full_kelly=full_kelly,
            fractional_kelly=fractional_kelly,
            position_size=position_size,
            max_loss=max_loss,
            risk_pct=risk_pct,
        )

    # ------------------------------------------------------------------
    # Continuous Kelly from EV / variance
    # ------------------------------------------------------------------

    def compute_kelly_from_ev(
        self, ev: float, variance: float, bankroll: float
    ) -> KellyResult:
        """
        Continuous Kelly criterion: f* = EV / variance.

        Useful when modeling PnL as a continuous process rather than
        discrete binary outcomes.

        Parameters
        ----------
        ev : float
            Expected value of the bet (per unit bankroll).
        variance : float
            Variance of returns (per unit bankroll).
        bankroll : float
            Current total bankroll.

        Returns
        -------
        KellyResult
            Sizing result. full_kelly carries the continuous fraction,
            fractional_kelly applies self.kelly_fraction, and position_size
            is clamped to max_position_pct * bankroll.
        """
        if variance <= 0.0:
            return KellyResult(
                full_kelly=0.0,
                fractional_kelly=0.0,
                position_size=0.0,
                max_loss=0.0,
                risk_pct=0.0,
            )

        full_kelly = ev / variance

        if full_kelly <= 0.0:
            return KellyResult(
                full_kelly=full_kelly,
                fractional_kelly=0.0,
                position_size=0.0,
                max_loss=0.0,
                risk_pct=0.0,
            )

        fractional_kelly = full_kelly * self.kelly_fraction

        raw_position = fractional_kelly * bankroll
        max_allowed = self.max_position_pct * bankroll
        position_size = min(raw_position, max_allowed)

        max_loss = position_size
        risk_pct = position_size / bankroll if bankroll > 0.0 else 0.0

        return KellyResult(
            full_kelly=full_kelly,
            fractional_kelly=fractional_kelly,
            position_size=position_size,
            max_loss=max_loss,
            risk_pct=risk_pct,
        )

    # ------------------------------------------------------------------
    # Portfolio Kelly (multi-asset)
    # ------------------------------------------------------------------

    def compute_portfolio_kelly(
        self, signals: list[tuple[float, float]], bankroll: float
    ) -> list[float]:
        """
        Multi-asset Kelly: size positions so they sum to at most 20% of bankroll.

        Each signal is (prob_win, decimal_odds).  Individual Kelly fractions are
        computed, normalised proportionally, and then scaled so the aggregate
        exposure does not exceed 20% of bankroll.

        Parameters
        ----------
        signals : list[tuple[float, float]]
            List of (prob_win, decimal_odds) pairs.
        bankroll : float
            Current total bankroll.

        Returns
        -------
        list[float]
            Dollar position sizes, one per signal (0.0 for invalid/negative-edge
            signals).
        """
        portfolio_cap = 0.20 * bankroll

        raw_fractions: list[float] = []
        for prob, odds in signals:
            if not self.validate_inputs(prob, odds):
                raw_fractions.append(0.0)
                continue

            b = odds - 1.0
            q = 1.0 - prob
            fk = (prob * b - q) / b
            if fk <= 0.0:
                raw_fractions.append(0.0)
            else:
                raw_fractions.append(fk * self.kelly_fraction)

        total_fraction = sum(raw_fractions)

        if total_fraction <= 0.0:
            return [0.0] * len(signals)

        # Scale so total dollar exposure == portfolio_cap (or less if already under)
        # Each position's share is proportional to its individual Kelly fraction.
        positions: list[float] = []
        for frac in raw_fractions:
            if frac <= 0.0:
                positions.append(0.0)
            else:
                # Proportional share of the portfolio cap
                dollar_size = (frac / total_fraction) * portfolio_cap
                positions.append(dollar_size)

        return positions

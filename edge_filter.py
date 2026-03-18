"""
edge_filter.py — HEISENBERG Project
EDGE_AGENT: Z-score spread, EV calculation, and net edge signal computation.
Python 3.11, numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpreadData:
    bid: float
    ask: float
    mid: float
    spread: float
    spread_bps: int


@dataclass
class EdgeSignal:
    z_score: float
    expected_value: float
    net_edge: float
    is_tradeable: bool
    confidence: float


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    return max(low, min(high, value))


class EdgeFilter:
    """
    Filters trade opportunities based on z-score spread, expected value,
    and a composite net edge signal.

    Parameters
    ----------
    min_edge_bps : int
        Minimum net edge in basis points required for a trade to be tradeable.
    min_z_score : float
        Minimum absolute z-score required for a trade to be tradeable.
    max_spread_bps : int
        Maximum allowable spread in basis points for a trade to be tradeable.
    """

    def __init__(
        self,
        min_edge_bps: int = 50,
        min_z_score: float = 1.5,
        max_spread_bps: int = 500,
    ) -> None:
        self.min_edge_bps = min_edge_bps
        self.min_z_score = min_z_score
        self.max_spread_bps = max_spread_bps

    def compute_z_score(
        self, price: float, prices_history: list[float], window: int = 60
    ) -> float:
        """
        Compute the rolling z-score of price relative to prices_history.

        Rolling z-score: (price - rolling_mean) / rolling_std

        Parameters
        ----------
        price : float
            The current price to score.
        prices_history : list[float]
            Historical prices used to compute the rolling mean and std.
        window : int
            Number of most-recent history entries to use.

        Returns
        -------
        float
            The z-score, or 0.0 if the window is too small or std is zero.
        """
        if not prices_history:
            return 0.0

        windowed: np.ndarray = np.array(prices_history[-window:], dtype=np.float64)

        if len(windowed) < 2:
            return 0.0

        mean: float = float(np.mean(windowed))
        std: float = float(np.std(windowed, ddof=1))

        if std == 0.0:
            return 0.0

        return float((price - mean) / std)

    def compute_ev(
        self,
        prob: float,
        odds_yes: float,
        odds_no: float,
        fee_bps: int = 20,
    ) -> float:
        """
        Compute expected value of a trade.

        EV = prob * odds_yes - (1 - prob) * odds_no - fee_bps / 10000

        Where:
            odds_yes = 1 / ask_price   (implied from market ask)
            odds_no  = 1 / (1 - bid_price)

        Parameters
        ----------
        prob : float
            Estimated probability the event resolves YES.
        odds_yes : float
            Decimal odds for YES side (1 / ask_price).
        odds_no : float
            Decimal odds for NO side (1 / (1 - bid_price)).
        fee_bps : int
            Trading fee in basis points.

        Returns
        -------
        float
            Expected value of the trade.
        """
        fee: float = fee_bps / 10_000.0
        return prob * odds_yes - (1.0 - prob) * odds_no - fee

    def compute_net_edge(
        self,
        z_score: float,
        ev: float,
        bayesian_posterior: float,
    ) -> float:
        """
        Combine z-score, EV, and Bayesian posterior into a single net edge signal.

        net_edge = 0.4 * clamp(z_score / 3, -1, 1)
                 + 0.4 * clamp(ev * 10, -1, 1)
                 + 0.2 * (2 * bayesian_posterior - 1)

        Parameters
        ----------
        z_score : float
            Rolling z-score of the current price.
        ev : float
            Expected value of the trade.
        bayesian_posterior : float
            Posterior probability estimate in [0, 1].

        Returns
        -------
        float
            Net edge in [-1, 1] range (approximately).
        """
        z_component = 0.4 * _clamp(z_score / 3.0, -1.0, 1.0)
        ev_component = 0.4 * _clamp(ev * 10.0, -1.0, 1.0)
        bayes_component = 0.2 * (2.0 * bayesian_posterior - 1.0)
        return z_component + ev_component + bayes_component

    def filter(
        self,
        spread: SpreadData,
        z_score: float,
        ev: float,
        posterior: float,
    ) -> EdgeSignal:
        """
        Evaluate whether a trade opportunity clears all edge thresholds.

        A trade is tradeable when ALL of the following hold:
            - spread.spread_bps <= max_spread_bps
            - abs(z_score) >= min_z_score
            - net_edge > min_edge_bps / 10000

        Confidence is defined as the absolute net edge (clamped to [0, 1]).

        Parameters
        ----------
        spread : SpreadData
            Current bid/ask/spread data for the market.
        z_score : float
            Rolling z-score of the current price.
        ev : float
            Expected value of the trade.
        posterior : float
            Bayesian posterior probability estimate in [0, 1].

        Returns
        -------
        EdgeSignal
            Full signal including tradeability flag and confidence score.
        """
        net_edge = self.compute_net_edge(z_score, ev, posterior)
        min_edge_threshold = self.min_edge_bps / 10_000.0

        is_tradeable: bool = (
            spread.spread_bps <= self.max_spread_bps
            and abs(z_score) >= self.min_z_score
            and net_edge > min_edge_threshold
        )

        confidence = float(_clamp(abs(net_edge), 0.0, 1.0))

        return EdgeSignal(
            z_score=z_score,
            expected_value=ev,
            net_edge=net_edge,
            is_tradeable=is_tradeable,
            confidence=confidence,
        )

    def compute_spread(self, bid: float, ask: float) -> SpreadData:
        """
        Compute spread metrics from bid and ask prices.

        Parameters
        ----------
        bid : float
            Best bid price (in [0, 1] for prediction markets).
        ask : float
            Best ask price (in [0, 1] for prediction markets).

        Returns
        -------
        SpreadData
            Spread metrics including mid, raw spread, and spread in bps.
        """
        mid: float = (bid + ask) / 2.0
        spread: float = ask - bid
        spread_bps: int = round(spread * 10_000)
        return SpreadData(
            bid=bid,
            ask=ask,
            mid=mid,
            spread=spread,
            spread_bps=spread_bps,
        )

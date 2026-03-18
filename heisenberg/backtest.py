"""
backtest.py — HEISENBERG project
Historical backtest engine using Polymarket public prices-history data.

Endpoint: https://clob.polymarket.com/prices-history
Params:   market (token_id), startTs (unix), endTs (unix), fidelity (int, minutes)

No live trading. No wallet. Read-only analysis only.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

import httpx

PRICES_HISTORY_URL = "https://clob.polymarket.com/prices-history"


# ---------------------------------------------------------------------------
# Configuration & result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    token_id: str
    start_ts: int
    end_ts: int
    fidelity_minutes: int = 5
    initial_capital: float = 1_000.0
    kelly_fraction: float = 0.25  # fractional Kelly multiplier


@dataclass
class Trade:
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    size: float          # position size in capital units
    pnl: float           # realised profit / loss
    edge_signal: float   # z-score / signal strength at entry


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """
    Mean-reversion backtest engine for a single Polymarket token.

    Signal logic
    ------------
    Rolling window of the last ``window`` price observations:
      - Enter LONG when  price < mean - 1.5 * std
      - Exit  when       price >= mean

    Position sizing
    ---------------
    Fractional Kelly (0.25x) applied to the estimated edge:
        edge  = (mean - entry_price)          # expected gain per unit
        odds  = mean / entry_price - 1        # fractional gain
        kelly = edge / mean                   # simplified Kelly fraction
        size  = capital * kelly_fraction * kelly
    """

    SIGNAL_THRESHOLD = 1.5   # standard deviations below mean to trigger entry
    ROLLING_WINDOW = 20      # number of periods for rolling statistics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, config: BacktestConfig) -> BacktestResult:
        """
        Fetch price history and run the full backtest.

        Parameters
        ----------
        config:
            A ``BacktestConfig`` describing the market and simulation params.

        Returns
        -------
        BacktestResult
            Aggregated performance metrics and individual trade records.
        """
        raw = self._fetch_prices(config)
        prices = self._validate_prices(raw)

        if len(prices) < 2:
            return BacktestResult()

        trades = self.simulate_trades(prices, config)

        equity_curve = self._build_equity_curve(trades, config.initial_capital)
        returns = self._equity_to_returns(equity_curve)

        total_pnl = sum(t.pnl for t in trades)
        wins = [t for t in trades if t.pnl > 0]
        win_rate = len(wins) / len(trades) if trades else 0.0

        sharpe = self.calculate_sharpe(returns)
        max_dd = self.calculate_max_drawdown(equity_curve)

        return BacktestResult(
            trades=trades,
            total_pnl=round(total_pnl, 6),
            sharpe_ratio=round(sharpe, 6),
            max_drawdown=round(max_dd, 6),
            win_rate=round(win_rate, 6),
            total_trades=len(trades),
        )

    # ------------------------------------------------------------------
    # Static / helper methods (public so tests can call them directly)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_sharpe(
        returns: list[float],
        periods_per_year: int = 105_120,  # 5-min periods in a year
    ) -> float:
        """
        Annualised Sharpe ratio (risk-free rate = 0).

        Parameters
        ----------
        returns:
            Period-over-period returns as decimal fractions (e.g., 0.01 = 1 %).
        periods_per_year:
            Number of periods in a year used for annualisation.
            Default is 105 120 = 365 * 24 * 60 / 5 (5-minute bars).

        Returns
        -------
        float
            Annualised Sharpe ratio. Returns 0.0 if fewer than 2 returns
            or if standard deviation is zero.
        """
        if len(returns) < 2:
            return 0.0

        mean_r = statistics.mean(returns)
        try:
            std_r = statistics.stdev(returns)
        except statistics.StatisticsError:
            return 0.0

        if std_r == 0.0:
            return 0.0

        return (mean_r / std_r) * math.sqrt(periods_per_year)

    @staticmethod
    def calculate_max_drawdown(equity_curve: list[float]) -> float:
        """
        Maximum peak-to-trough drawdown of an equity curve.

        Parameters
        ----------
        equity_curve:
            Sequence of equity values (e.g., [1000, 1050, 980, ...]).

        Returns
        -------
        float
            Max drawdown as a positive decimal fraction (e.g., 0.10 = 10 %).
            Returns 0.0 for curves with fewer than 2 points.
        """
        if len(equity_curve) < 2:
            return 0.0

        peak = equity_curve[0]
        max_dd = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def simulate_trades(
        self,
        prices: list[dict],
        config: BacktestConfig | None = None,
    ) -> list[Trade]:
        """
        Simulate mean-reversion trades over a validated price series.

        Parameters
        ----------
        prices:
            List of dicts with keys ``t`` (unix timestamp) and ``p`` (float price).
        config:
            Optional ``BacktestConfig``; used for ``kelly_fraction`` and
            ``initial_capital``. Defaults to fraction=0.25, capital=1 000.

        Returns
        -------
        list[Trade]
            All completed (entry + exit) trades.
        """
        kelly_fraction = config.kelly_fraction if config else 0.25
        capital = config.initial_capital if config else 1_000.0

        trades: list[Trade] = []
        in_position = False
        entry_ts: int = 0
        entry_price: float = 0.0
        position_size: float = 0.0
        edge_signal: float = 0.0

        window = self.ROLLING_WINDOW

        for i, bar in enumerate(prices):
            ts = bar["t"]
            price = bar["p"]

            # Need at least ``window`` data points for rolling stats
            if i < window:
                continue

            window_prices = [prices[j]["p"] for j in range(i - window, i)]
            roll_mean = statistics.mean(window_prices)

            if len(window_prices) < 2:
                continue

            try:
                roll_std = statistics.stdev(window_prices)
            except statistics.StatisticsError:
                continue

            if roll_std == 0.0:
                continue

            z_score = (price - roll_mean) / roll_std  # negative = below mean

            if not in_position:
                # Entry condition: price is more than threshold stds below mean
                if z_score < -self.SIGNAL_THRESHOLD:
                    # Simplified Kelly: edge / mean (bounded 0..1)
                    edge = (roll_mean - price) / roll_mean
                    kelly_raw = max(0.0, min(edge, 1.0))
                    position_size = capital * kelly_fraction * kelly_raw

                    if position_size > 0:
                        in_position = True
                        entry_ts = ts
                        entry_price = price
                        edge_signal = z_score
            else:
                # Exit condition: price has reverted to rolling mean
                if price >= roll_mean:
                    pnl = position_size * (price - entry_price) / entry_price
                    trades.append(
                        Trade(
                            entry_ts=entry_ts,
                            exit_ts=ts,
                            entry_price=entry_price,
                            exit_price=price,
                            size=position_size,
                            pnl=round(pnl, 6),
                            edge_signal=round(edge_signal, 6),
                        )
                    )
                    # Reinvest capital
                    capital = max(0.0, capital + pnl)
                    in_position = False

        return trades

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_prices(self, config: BacktestConfig) -> list[dict]:
        """
        Synchronous HTTP GET to the Polymarket prices-history endpoint.

        Returns
        -------
        list[dict]
            Raw list of price-bar objects returned by the API.

        Raises
        ------
        httpx.HTTPStatusError
            On non-2xx responses.
        ValueError
            If the response body is missing the expected ``history`` key.
        """
        params: dict[str, Any] = {
            "market": config.token_id,
            "startTs": config.start_ts,
            "endTs": config.end_ts,
            "fidelity": config.fidelity_minutes,
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.get(PRICES_HISTORY_URL, params=params)
            response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict) or "history" not in data:
            raise ValueError(
                f"Unexpected API response structure. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )

        return data["history"]  # list of {t: int, p: float}

    @staticmethod
    def _validate_prices(raw: list[dict]) -> list[dict]:
        """
        Strip bars with missing or non-finite prices.

        Parameters
        ----------
        raw:
            Raw list from the API.

        Returns
        -------
        list[dict]
            Cleaned list, sorted by timestamp ascending.
        """
        cleaned: list[dict] = []
        for bar in raw:
            try:
                t = int(bar["t"])
                p = float(bar["p"])
            except (KeyError, TypeError, ValueError):
                continue

            if not math.isfinite(p) or p <= 0:
                continue

            cleaned.append({"t": t, "p": p})

        # Sort by timestamp in case the API returns out-of-order bars
        cleaned.sort(key=lambda b: b["t"])
        return cleaned

    @staticmethod
    def _build_equity_curve(
        trades: list[Trade],
        initial_capital: float,
    ) -> list[float]:
        """
        Build a cumulative equity curve from completed trades.
        """
        equity = initial_capital
        curve = [equity]
        for trade in trades:
            equity += trade.pnl
            curve.append(max(0.0, equity))
        return curve

    @staticmethod
    def _equity_to_returns(equity_curve: list[float]) -> list[float]:
        """
        Convert an equity curve into period-over-period decimal returns.
        """
        returns: list[float] = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            curr = equity_curve[i]
            if prev > 0:
                returns.append((curr - prev) / prev)
            else:
                returns.append(0.0)
        return returns

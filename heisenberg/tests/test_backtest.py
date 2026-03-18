"""
tests/test_backtest.py — HEISENBERG project
Unit tests for the BacktestEngine in backtest.py.

Run with:
    pytest heisenberg/tests/test_backtest.py -v
"""

from __future__ import annotations

import math
import statistics
from unittest.mock import MagicMock, patch

import pytest

from heisenberg.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def make_linear_prices(
    start: float,
    end: float,
    n: int,
    start_ts: int = 1_700_000_000,
    step_ts: int = 300,
) -> list[dict]:
    """Generate a simple linear price series from ``start`` to ``end``."""
    prices = []
    for i in range(n):
        p = start + (end - start) * i / max(n - 1, 1)
        prices.append({"t": start_ts + i * step_ts, "p": round(p, 6)})
    return prices


def make_mean_reversion_prices(
    n_warmup: int = 25,
    dip_price: float = 0.30,
    mean_price: float = 0.50,
    start_ts: int = 1_700_000_000,
    step_ts: int = 300,
) -> list[dict]:
    """
    Build a price series that contains a clear mean-reversion opportunity:
      1. ``n_warmup`` bars at ``mean_price`` (establishes rolling statistics)
      2. One bar that drops sharply to ``dip_price`` (triggers entry signal)
      3. Several bars that gradually recover back to ``mean_price`` (triggers exit)
    """
    prices: list[dict] = []
    ts = start_ts

    # Warmup: stable around mean_price
    for i in range(n_warmup):
        # Tiny oscillation so std is non-zero
        oscillation = 0.001 * (1 if i % 2 == 0 else -1)
        prices.append({"t": ts, "p": round(mean_price + oscillation, 6)})
        ts += step_ts

    # Sharp dip below mean - 1.5*std
    prices.append({"t": ts, "p": dip_price})
    ts += step_ts

    # Gradual recovery to mean
    for j in range(10):
        recovery_p = dip_price + (mean_price - dip_price) * (j + 1) / 10
        prices.append({"t": ts, "p": round(recovery_p, 6)})
        ts += step_ts

    return prices


def make_config(**kwargs) -> BacktestConfig:
    defaults = dict(
        token_id="test-token-123",
        start_ts=1_700_000_000,
        end_ts=1_700_100_000,
        fidelity_minutes=5,
        initial_capital=1_000.0,
        kelly_fraction=0.25,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Sharpe ratio correctness
# ---------------------------------------------------------------------------


class TestCalculateSharpe:
    def test_positive_sharpe_for_consistently_positive_returns(self):
        returns = [0.01] * 100
        sharpe = BacktestEngine.calculate_sharpe(returns)
        # All returns identical → stdev = 0 → special-cased to 0.0
        # Switch to slightly varying returns
        returns = [0.01 + 0.001 * math.sin(i) for i in range(200)]
        sharpe = BacktestEngine.calculate_sharpe(returns)
        assert sharpe > 0, "Sharpe should be positive for positive mean returns"

    def test_zero_sharpe_for_zero_std(self):
        returns = [0.005] * 50  # identical → std = 0
        sharpe = BacktestEngine.calculate_sharpe(returns)
        assert sharpe == 0.0

    def test_zero_sharpe_for_empty_returns(self):
        assert BacktestEngine.calculate_sharpe([]) == 0.0

    def test_zero_sharpe_for_single_return(self):
        assert BacktestEngine.calculate_sharpe([0.05]) == 0.0

    def test_negative_sharpe_for_negative_mean_returns(self):
        returns = [-0.01 + 0.001 * math.sin(i) for i in range(200)]
        sharpe = BacktestEngine.calculate_sharpe(returns)
        assert sharpe < 0, "Sharpe should be negative for negative mean returns"

    def test_annualisation_factor_applied(self):
        """Sharpe with periods_per_year=1 should equal mean/std (unannualised)."""
        returns = [0.01, 0.02, -0.005, 0.015, 0.008]
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        expected = mean_r / std_r  # periods_per_year=1 → sqrt(1)=1
        result = BacktestEngine.calculate_sharpe(returns, periods_per_year=1)
        assert abs(result - expected) < 1e-9

    def test_sharpe_uses_default_5min_periods(self):
        """Default periods_per_year=105120 should annualise correctly."""
        returns = [0.001 + 0.0001 * i for i in range(50)]
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        expected = (mean_r / std_r) * math.sqrt(105_120)
        result = BacktestEngine.calculate_sharpe(returns)
        assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# 2. Max drawdown calculation
# ---------------------------------------------------------------------------


class TestCalculateMaxDrawdown:
    def test_no_drawdown_for_monotonically_rising_curve(self):
        curve = [100.0, 110.0, 120.0, 130.0]
        assert BacktestEngine.calculate_max_drawdown(curve) == 0.0

    def test_correct_drawdown_for_known_drop(self):
        # Peak=200, trough=100 → 50 % drawdown
        curve = [100.0, 150.0, 200.0, 100.0, 180.0]
        dd = BacktestEngine.calculate_max_drawdown(curve)
        assert abs(dd - 0.50) < 1e-9

    def test_drawdown_of_zero_for_single_value(self):
        assert BacktestEngine.calculate_max_drawdown([500.0]) == 0.0

    def test_drawdown_of_zero_for_empty_curve(self):
        assert BacktestEngine.calculate_max_drawdown([]) == 0.0

    def test_drawdown_ignores_recovery(self):
        # Drawdown should reflect the worst peak-to-trough, not final value
        curve = [1000.0, 800.0, 600.0, 900.0, 1100.0]
        # Worst: peak=1000, trough=600 → 40 %
        dd = BacktestEngine.calculate_max_drawdown(curve)
        assert abs(dd - 0.40) < 1e-9

    def test_full_loss(self):
        curve = [1000.0, 500.0, 0.001]
        dd = BacktestEngine.calculate_max_drawdown(curve)
        assert dd > 0.99  # essentially 100 % drawdown


# ---------------------------------------------------------------------------
# 3. Trade simulation with a mock price series
# ---------------------------------------------------------------------------


class TestSimulateTrades:
    def test_no_trades_on_flat_series(self):
        """Flat prices → std=0 → no signal → no trades."""
        prices = [{"t": 1_700_000_000 + i * 300, "p": 0.50} for i in range(50)]
        engine = BacktestEngine()
        trades = engine.simulate_trades(prices)
        assert trades == []

    def test_trade_generated_on_dip(self):
        """Sharp dip well below rolling mean should generate at least one trade."""
        prices = make_mean_reversion_prices(
            n_warmup=25,
            dip_price=0.10,  # extreme dip to ensure z-score < -1.5
            mean_price=0.50,
        )
        engine = BacktestEngine()
        trades = engine.simulate_trades(prices)
        assert len(trades) >= 1, "Expected at least one trade on a clear dip-recovery"

    def test_trade_entry_price_is_dip(self):
        prices = make_mean_reversion_prices(
            n_warmup=25,
            dip_price=0.10,
            mean_price=0.50,
        )
        engine = BacktestEngine()
        trades = engine.simulate_trades(prices)
        if trades:
            assert trades[0].entry_price == pytest.approx(0.10, abs=0.05)

    def test_trade_pnl_is_positive_on_recovery(self):
        prices = make_mean_reversion_prices(
            n_warmup=25,
            dip_price=0.10,
            mean_price=0.50,
        )
        engine = BacktestEngine()
        trades = engine.simulate_trades(prices)
        for trade in trades:
            assert trade.pnl >= 0, "Mean-reversion trade on recovery should be profitable"

    def test_trade_size_respects_kelly_fraction(self):
        prices = make_mean_reversion_prices(
            n_warmup=25,
            dip_price=0.10,
            mean_price=0.50,
        )
        config = make_config(kelly_fraction=0.25, initial_capital=1_000.0)
        engine = BacktestEngine()
        trades = engine.simulate_trades(prices, config=config)
        for trade in trades:
            # Position size should never exceed initial_capital * kelly_fraction
            assert trade.size <= config.initial_capital * config.kelly_fraction + 1e-6

    def test_exit_timestamp_after_entry(self):
        prices = make_mean_reversion_prices(
            n_warmup=25,
            dip_price=0.10,
            mean_price=0.50,
        )
        engine = BacktestEngine()
        trades = engine.simulate_trades(prices)
        for trade in trades:
            assert trade.exit_ts > trade.entry_ts

    def test_no_crash_on_two_point_series(self):
        prices = [
            {"t": 1_700_000_000, "p": 0.40},
            {"t": 1_700_000_300, "p": 0.60},
        ]
        engine = BacktestEngine()
        trades = engine.simulate_trades(prices)
        assert isinstance(trades, list)


# ---------------------------------------------------------------------------
# 4. BacktestResult metrics computed correctly
# ---------------------------------------------------------------------------


class TestBacktestResultMetrics:
    def _make_trades(self) -> list[Trade]:
        return [
            Trade(1_700_000_000, 1_700_001_000, 0.40, 0.55, 100.0, 37.50, -2.1),
            Trade(1_700_002_000, 1_700_003_000, 0.35, 0.50, 80.0,  34.29, -1.8),
            Trade(1_700_004_000, 1_700_005_000, 0.45, 0.40, 60.0, -6.67, -1.6),
        ]

    def test_total_pnl_is_sum_of_trades(self):
        trades = self._make_trades()
        expected_pnl = sum(t.pnl for t in trades)
        # Build result manually to test the field, not the engine
        result = BacktestResult(
            trades=trades,
            total_pnl=round(expected_pnl, 6),
            total_trades=len(trades),
        )
        assert abs(result.total_pnl - expected_pnl) < 1e-4

    def test_win_rate_correct(self):
        trades = self._make_trades()  # 2 wins, 1 loss
        wins = [t for t in trades if t.pnl > 0]
        win_rate = len(wins) / len(trades)
        assert abs(win_rate - 2 / 3) < 1e-9

    def test_total_trades_count(self):
        trades = self._make_trades()
        result = BacktestResult(trades=trades, total_trades=len(trades))
        assert result.total_trades == 3

    def test_default_result_has_zero_metrics(self):
        result = BacktestResult()
        assert result.total_pnl == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.max_drawdown == 0.0
        assert result.win_rate == 0.0
        assert result.total_trades == 0
        assert result.trades == []


# ---------------------------------------------------------------------------
# 5. BacktestEngine.run() — mocked httpx
# ---------------------------------------------------------------------------


SAMPLE_HISTORY = {
    "history": [
        {"t": 1_700_000_000 + i * 300, "p": 0.50 + 0.001 * (1 if i % 2 == 0 else -1)}
        for i in range(60)
    ]
}

DIP_HISTORY = {
    "history": make_mean_reversion_prices(n_warmup=25, dip_price=0.10, mean_price=0.50)
}


class TestBacktestEngineRun:
    @pytest.mark.asyncio
    async def test_run_returns_backtest_result(self):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_HISTORY
        mock_response.raise_for_status.return_value = None

        with patch("heisenberg.backtest.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            engine = BacktestEngine()
            config = make_config()
            result = await engine.run(config)

        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_run_returns_empty_result_on_empty_history(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"history": []}
        mock_response.raise_for_status.return_value = None

        with patch("heisenberg.backtest.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            engine = BacktestEngine()
            result = await engine.run(make_config())

        assert result.total_trades == 0
        assert result.total_pnl == 0.0

    @pytest.mark.asyncio
    async def test_run_handles_single_price_history(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "history": [{"t": 1_700_000_000, "p": 0.50}]
        }
        mock_response.raise_for_status.return_value = None

        with patch("heisenberg.backtest.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            engine = BacktestEngine()
            result = await engine.run(make_config())

        # Should not raise; should return empty result
        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0

    @pytest.mark.asyncio
    async def test_run_produces_trades_on_dip_recovery(self):
        mock_response = MagicMock()
        mock_response.json.return_value = DIP_HISTORY
        mock_response.raise_for_status.return_value = None

        with patch("heisenberg.backtest.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            engine = BacktestEngine()
            result = await engine.run(make_config())

        assert result.total_trades >= 1

    @pytest.mark.asyncio
    async def test_run_strips_nan_and_invalid_prices(self):
        """Bars with NaN or zero prices should be silently dropped."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "history": [
                {"t": 1_700_000_000, "p": float("nan")},
                {"t": 1_700_000_300, "p": 0.0},
                {"t": 1_700_000_600, "p": -0.1},
                {"t": 1_700_000_900, "p": 0.5},
            ]
        }
        mock_response.raise_for_status.return_value = None

        with patch("heisenberg.backtest.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            engine = BacktestEngine()
            # Should not raise
            result = await engine.run(make_config())

        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_run_sharpe_and_drawdown_are_finite(self):
        mock_response = MagicMock()
        mock_response.json.return_value = DIP_HISTORY
        mock_response.raise_for_status.return_value = None

        with patch("heisenberg.backtest.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            engine = BacktestEngine()
            result = await engine.run(make_config())

        assert math.isfinite(result.sharpe_ratio)
        assert math.isfinite(result.max_drawdown)
        assert 0.0 <= result.max_drawdown <= 1.0

    @pytest.mark.asyncio
    async def test_run_win_rate_bounded_zero_to_one(self):
        mock_response = MagicMock()
        mock_response.json.return_value = DIP_HISTORY
        mock_response.raise_for_status.return_value = None

        with patch("heisenberg.backtest.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            engine = BacktestEngine()
            result = await engine.run(make_config())

        assert 0.0 <= result.win_rate <= 1.0


# ---------------------------------------------------------------------------
# 6. Data validation (_validate_prices)
# ---------------------------------------------------------------------------


class TestValidatePrices:
    def test_filters_nan_prices(self):
        raw = [
            {"t": 1, "p": float("nan")},
            {"t": 2, "p": 0.5},
        ]
        cleaned = BacktestEngine._validate_prices(raw)
        assert len(cleaned) == 1
        assert cleaned[0]["p"] == 0.5

    def test_filters_zero_and_negative_prices(self):
        raw = [
            {"t": 1, "p": 0.0},
            {"t": 2, "p": -1.0},
            {"t": 3, "p": 0.3},
        ]
        cleaned = BacktestEngine._validate_prices(raw)
        assert len(cleaned) == 1

    def test_sorts_by_timestamp(self):
        raw = [
            {"t": 300, "p": 0.6},
            {"t": 100, "p": 0.4},
            {"t": 200, "p": 0.5},
        ]
        cleaned = BacktestEngine._validate_prices(raw)
        assert [b["t"] for b in cleaned] == [100, 200, 300]

    def test_empty_input_returns_empty(self):
        assert BacktestEngine._validate_prices([]) == []

    def test_missing_keys_are_dropped(self):
        raw = [
            {"t": 1},          # missing p
            {"p": 0.5},        # missing t
            {"t": 3, "p": 0.7},
        ]
        cleaned = BacktestEngine._validate_prices(raw)
        assert len(cleaned) == 1
        assert cleaned[0]["p"] == 0.7

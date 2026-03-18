"""
test_bot.py — Unit tests for HEISENBERG bot.py orchestrator.

Tests the pipeline wiring: each module is mocked so we verify
the orchestrator calls components in the correct order with
correct data flow — not re-testing the modules themselves.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Helpers — minimal stubs matching real dataclasses
# ---------------------------------------------------------------------------

def make_order_book(bid=0.45, ask=0.55, bid_size=100.0, ask_size=120.0):
    """Return a mock OrderBook with controllable bid/ask."""
    book = MagicMock()
    book.best_bid = bid
    book.best_ask = ask
    book.mid_price = (bid + ask) / 2
    bid_level = MagicMock(); bid_level.price = bid; bid_level.size = bid_size
    ask_level = MagicMock(); ask_level.price = ask; ask_level.size = ask_size
    book.bids = [bid_level]
    book.asks = [ask_level]
    return book


def make_market_info(question="Will BTC be above 70k in 5 min?", token_ids=("tok_yes", "tok_no")):
    m = MagicMock()
    m.question = question
    m.tokens = [{"token_id": t} for t in token_ids]
    return m


# ---------------------------------------------------------------------------
# Import bot under test
# ---------------------------------------------------------------------------

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot import HeisenbergBot, PipelineSignal, BANKROLL, KELLY_FRACTION


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineSignalSummary:
    """PipelineSignal.summary() output format."""

    def _make_signal(self, is_tradeable: bool, net_edge: float) -> PipelineSignal:
        spread = MagicMock(); spread.spread_bps = 100
        edge = MagicMock()
        edge.is_tradeable = is_tradeable
        edge.z_score = 2.1
        edge.expected_value = 0.03
        edge.net_edge = net_edge
        return PipelineSignal(
            token_id="tok_yes",
            market_question="Will BTC be above 70k?",
            mid_price=0.50,
            spread_data=spread,
            edge_signal=edge,
            kelly_position_size=25.0,
            reservation_price=0.499,
            bid_quote=0.48,
            ask_quote=0.52,
        )

    def test_tradeable_signal_shows_signal_label(self):
        s = self._make_signal(is_tradeable=True, net_edge=0.1)
        assert "SIGNAL" in s.summary()

    def test_non_tradeable_shows_skip_label(self):
        s = self._make_signal(is_tradeable=False, net_edge=0.01)
        assert "SKIP" in s.summary()

    def test_positive_edge_shows_buy(self):
        s = self._make_signal(is_tradeable=True, net_edge=0.15)
        assert "BUY" in s.summary()

    def test_negative_edge_shows_sell(self):
        s = self._make_signal(is_tradeable=False, net_edge=-0.05)
        assert "SELL" in s.summary()

    def test_summary_contains_mid_price(self):
        s = self._make_signal(is_tradeable=True, net_edge=0.1)
        assert "0.500" in s.summary()


class TestHeisenbergBotInit:
    """Bot initialises with correct default components."""

    def test_default_bankroll(self):
        bot = HeisenbergBot()
        assert bot.bankroll == BANKROLL

    def test_custom_bankroll(self):
        bot = HeisenbergBot(bankroll=5000.0)
        assert bot.bankroll == 5000.0

    def test_components_are_instantiated(self):
        bot = HeisenbergBot()
        assert bot.client is not None
        assert bot.bayesian is not None
        assert bot.edge_filter is not None
        assert bot.kelly is not None
        assert bot.stoikov is not None

    def test_price_history_starts_empty(self):
        bot = HeisenbergBot()
        assert bot._price_history == {}


class TestProcessToken:
    """_process_token: pipeline wiring and data flow."""

    @pytest.fixture
    def bot(self):
        return HeisenbergBot(bankroll=1000.0)

    @pytest.mark.asyncio
    async def test_returns_pipeline_signal_on_success(self, bot):
        book = make_order_book(bid=0.44, ask=0.56)
        bot.client.fetch_orderbook = AsyncMock(return_value=book)

        result = await bot._process_token("tok_yes", "BTC > 70k?", elapsed_t=60.0)

        assert isinstance(result, PipelineSignal)
        assert result.token_id == "tok_yes"
        assert result.market_question == "BTC > 70k?"

    @pytest.mark.asyncio
    async def test_returns_none_on_fetch_failure(self, bot):
        bot.client.fetch_orderbook = AsyncMock(side_effect=Exception("network error"))

        result = await bot._process_token("tok_yes", "BTC > 70k?", elapsed_t=0.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_book(self, bot):
        book = MagicMock()
        book.best_bid = None
        book.best_ask = None
        bot.client.fetch_orderbook = AsyncMock(return_value=book)

        result = await bot._process_token("tok_yes", "BTC > 70k?", elapsed_t=0.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_price_history_accumulates(self, bot):
        book = make_order_book(bid=0.44, ask=0.56)
        bot.client.fetch_orderbook = AsyncMock(return_value=book)

        for _ in range(5):
            await bot._process_token("tok_yes", "BTC > 70k?", elapsed_t=0.0)

        assert len(bot._price_history["tok_yes"]) == 5

    @pytest.mark.asyncio
    async def test_price_history_capped_at_200(self, bot):
        book = make_order_book(bid=0.44, ask=0.56)
        bot.client.fetch_orderbook = AsyncMock(return_value=book)

        for _ in range(250):
            await bot._process_token("tok_yes", "BTC > 70k?", elapsed_t=0.0)

        assert len(bot._price_history["tok_yes"]) == 200

    @pytest.mark.asyncio
    async def test_mid_price_recorded_correctly(self, bot):
        book = make_order_book(bid=0.40, ask=0.60)  # mid = 0.50
        bot.client.fetch_orderbook = AsyncMock(return_value=book)

        result = await bot._process_token("tok_yes", "BTC > 70k?", elapsed_t=0.0)

        assert result is not None
        assert abs(result.mid_price - 0.50) < 1e-9

    @pytest.mark.asyncio
    async def test_kelly_position_size_is_non_negative(self, bot):
        book = make_order_book(bid=0.44, ask=0.56)
        bot.client.fetch_orderbook = AsyncMock(return_value=book)

        result = await bot._process_token("tok_yes", "BTC > 70k?", elapsed_t=0.0)

        assert result is not None
        assert result.kelly_position_size >= 0.0

    @pytest.mark.asyncio
    async def test_quotes_in_valid_range(self, bot):
        book = make_order_book(bid=0.44, ask=0.56)
        bot.client.fetch_orderbook = AsyncMock(return_value=book)

        result = await bot._process_token("tok_yes", "BTC > 70k?", elapsed_t=0.0)

        assert result is not None
        assert 0.0 < result.bid_quote < 1.0
        assert 0.0 < result.ask_quote < 1.0
        assert result.bid_quote < result.ask_quote


class TestRunCycle:
    """run_cycle: market discovery → per-token processing."""

    @pytest.fixture
    def bot(self):
        return HeisenbergBot(bankroll=1000.0)

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_markets(self, bot):
        bot._fetch_active_btc_markets = AsyncMock(return_value=[])
        signals = await bot.run_cycle(1)
        assert signals == []

    @pytest.mark.asyncio
    async def test_returns_signals_for_found_markets(self, bot):
        market = make_market_info(token_ids=("tok_yes",))
        bot._fetch_active_btc_markets = AsyncMock(return_value=[market])

        book = make_order_book()
        bot.client.fetch_orderbook = AsyncMock(return_value=book)

        signals = await bot.run_cycle(1)
        assert len(signals) == 1
        assert isinstance(signals[0], PipelineSignal)

    @pytest.mark.asyncio
    async def test_skips_tokens_with_no_token_id(self, bot):
        market = MagicMock()
        market.question = "Test?"
        market.tokens = [{"token_id": ""}]  # empty token_id
        bot._fetch_active_btc_markets = AsyncMock(return_value=[market])

        signals = await bot.run_cycle(1)
        assert signals == []

    @pytest.mark.asyncio
    async def test_handles_mixed_success_and_failure(self, bot):
        market = make_market_info(token_ids=("tok_ok", "tok_fail"))
        bot._fetch_active_btc_markets = AsyncMock(return_value=[market])

        good_book = make_order_book()

        async def fetch_side_effect(token_id):
            if token_id == "tok_ok":
                return good_book
            raise Exception("bad token")

        bot.client.fetch_orderbook = AsyncMock(side_effect=fetch_side_effect)

        signals = await bot.run_cycle(1)
        assert len(signals) == 1
        assert signals[0].token_id == "tok_ok"


class TestFetchActiveBtcMarkets:
    """_fetch_active_btc_markets: wraps client and caps results."""

    @pytest.fixture
    def bot(self):
        return HeisenbergBot()

    @pytest.mark.asyncio
    async def test_returns_markets_on_success(self, bot):
        markets = [make_market_info() for _ in range(3)]
        bot.client.fetch_btc_5min_markets = AsyncMock(return_value=markets)

        result = await bot._fetch_active_btc_markets()
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_caps_at_max_markets(self, bot):
        from bot import MAX_MARKETS_PER_CYCLE
        markets = [make_market_info() for _ in range(MAX_MARKETS_PER_CYCLE + 5)]
        bot.client.fetch_btc_5min_markets = AsyncMock(return_value=markets)

        result = await bot._fetch_active_btc_markets()
        assert len(result) == MAX_MARKETS_PER_CYCLE

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_exception(self, bot):
        bot.client.fetch_btc_5min_markets = AsyncMock(side_effect=Exception("API down"))

        result = await bot._fetch_active_btc_markets()
        assert result == []

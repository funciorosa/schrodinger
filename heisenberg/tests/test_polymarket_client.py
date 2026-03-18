"""
tests/test_polymarket_client.py — Unit tests for PolymarketCLOBClient.

Uses pytest + pytest-asyncio with mocked httpx.AsyncClient so that no real
network calls are made.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from heisenberg.polymarket_client import (
    CLOBTick,
    MarketInfo,
    OrderBook,
    PriceLevel,
    PolymarketCLOBClient,
    _parse_price_levels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(
    status_code: int = 200,
    json_body: Any = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_body or {}
    response.headers = headers or {}
    response.request = MagicMock(spec=httpx.Request)

    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=response.request,
            response=response,
        )
    else:
        response.raise_for_status.return_value = None

    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> PolymarketCLOBClient:
    """Return a PolymarketCLOBClient with fast retry settings for tests."""
    return PolymarketCLOBClient(
        base_url="https://clob.polymarket.com",
        timeout=5.0,
        max_retries=2,
        backoff_base=0.01,   # tiny backoff so tests stay fast
    )


@pytest.fixture
def mock_http_get(client: PolymarketCLOBClient):
    """Patch the underlying AsyncClient.get method and yield the mock."""
    mock_async_client = AsyncMock(spec=httpx.AsyncClient)
    mock_async_client.is_closed = False

    with patch.object(client, "_get_client", return_value=mock_async_client):
        yield mock_async_client.get


# ---------------------------------------------------------------------------
# 1. fetch_orderbook — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_orderbook_returns_correct_structure(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """fetch_orderbook should parse bids and asks into PriceLevel objects."""
    token_id = "0xdeadbeef"
    raw_body = {
        "bids": [{"price": "0.55", "size": "100"}, {"price": "0.54", "size": "200"}],
        "asks": [{"price": "0.57", "size": "150"}],
    }
    mock_http_get.return_value = _make_response(200, raw_body)

    book = await client.fetch_orderbook(token_id)

    assert isinstance(book, OrderBook)
    assert book.token_id == token_id
    assert len(book.bids) == 2
    assert len(book.asks) == 1
    assert book.bids[0] == PriceLevel(price=0.55, size=100.0)
    assert book.bids[1] == PriceLevel(price=0.54, size=200.0)
    assert book.asks[0] == PriceLevel(price=0.57, size=150.0)


# ---------------------------------------------------------------------------
# 2. OrderBook computed properties
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_orderbook_best_bid_ask(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """best_bid, best_ask, and mid_price should be computed correctly."""
    raw_body = {
        "bids": [{"price": "0.50", "size": "10"}, {"price": "0.52", "size": "20"}],
        "asks": [{"price": "0.56", "size": "30"}, {"price": "0.60", "size": "5"}],
    }
    mock_http_get.return_value = _make_response(200, raw_body)

    book = await client.fetch_orderbook("0xabc")

    assert book.best_bid == pytest.approx(0.52)
    assert book.best_ask == pytest.approx(0.56)
    assert book.mid_price == pytest.approx(0.54)
    assert book.spread == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# 3. Empty order book edge case
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_orderbook_empty_book(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """An empty order book should yield None for best_bid/ask/mid/spread."""
    mock_http_get.return_value = _make_response(200, {"bids": [], "asks": []})

    book = await client.fetch_orderbook("0xempty")

    assert book.bids == []
    assert book.asks == []
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.mid_price is None
    assert book.spread is None


# ---------------------------------------------------------------------------
# 4. fetch_market_info — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_market_info_returns_correct_marketinfo(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """fetch_market_info should return a populated MarketInfo dataclass."""
    condition_id = "0xcondition1"
    raw_market = {
        "condition_id": condition_id,
        "question": "Will BTC close above $70k on 2026-01-01?",
        "end_date_iso": "2026-01-01T00:00:00Z",
        "volume": "1500000.0",
        "active": True,
        "tokens": [{"token_id": "0xtok1", "outcome": "Yes"}],
    }
    mock_http_get.return_value = _make_response(200, {"data": [raw_market]})

    info = await client.fetch_market_info(condition_id)

    assert isinstance(info, MarketInfo)
    assert info.condition_id == condition_id
    assert "BTC" in info.question
    assert info.end_date == "2026-01-01T00:00:00Z"
    assert info.volume == pytest.approx(1_500_000.0)
    assert info.active is True
    assert len(info.tokens) == 1


# ---------------------------------------------------------------------------
# 5. fetch_market_info — not found
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_market_info_not_found(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """fetch_market_info should return None when no market matches."""
    mock_http_get.return_value = _make_response(200, {"data": []})

    result = await client.fetch_market_info("0xnonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# 6. BTC market search filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_btc_markets_filters_active_only(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """fetch_btc_markets should exclude inactive markets."""
    raw_items = [
        {
            "condition_id": "0x01",
            "question": "Will BTC reach $100k?",
            "end_date_iso": None,
            "volume": "50000",
            "active": True,
            "tokens": [],
        },
        {
            "condition_id": "0x02",
            "question": "BTC above $50k? (closed)",
            "end_date_iso": "2025-01-01T00:00:00Z",
            "volume": "10000",
            "active": False,
            "tokens": [],
        },
    ]
    mock_http_get.return_value = _make_response(200, {"data": raw_items})

    markets = await client.fetch_btc_markets()

    assert len(markets) == 1
    assert markets[0].condition_id == "0x01"
    assert markets[0].active is True


# ---------------------------------------------------------------------------
# 7. Retry logic — 429 rate-limit response
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_triggers_on_429(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """The client should retry and eventually succeed after a 429 response."""
    token_id = "0xretry"
    success_body = {"bids": [{"price": "0.50", "size": "1"}], "asks": []}

    rate_limit_response = _make_response(429, {})
    rate_limit_response.raise_for_status.return_value = None   # don't raise on 429
    success_response = _make_response(200, success_body)

    mock_http_get.side_effect = [rate_limit_response, success_response]

    with patch("heisenberg.polymarket_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        book = await client.fetch_orderbook(token_id)

    # Should have slept once (rate-limit backoff) and then succeeded
    assert mock_sleep.call_count >= 1
    assert isinstance(book, OrderBook)
    assert len(book.bids) == 1


# ---------------------------------------------------------------------------
# 8. Retry logic — 500 server error exhaustion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_exhausted_on_500_raises(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """After all retries are exhausted on 500 errors, an exception is raised."""
    server_error = _make_response(500, {})
    server_error.raise_for_status.return_value = None   # we handle 500 manually

    # max_retries=2 means 3 total attempts
    mock_http_get.side_effect = [server_error, server_error, server_error]

    with patch("heisenberg.polymarket_client.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(httpx.HTTPStatusError):
            await client.fetch_orderbook("0x500token")

    assert mock_http_get.call_count == 3


# ---------------------------------------------------------------------------
# 9. Price parsing — malformed / missing fields
# ---------------------------------------------------------------------------

def test_parse_price_levels_handles_malformed_entries() -> None:
    """_parse_price_levels should skip bad entries without crashing."""
    raw = [
        {"price": "0.60", "size": "100"},
        {"price": "not_a_number", "size": "50"},   # bad price
        {"size": "50"},                              # missing price key
        {},                                          # completely empty
        {"price": "0.40", "size": "abc"},            # bad size
        {"price": "0.30", "size": "200"},
    ]
    levels = _parse_price_levels(raw)
    # Only the two fully valid entries should parse
    assert len(levels) == 2
    assert levels[0] == PriceLevel(price=0.60, size=100.0)
    assert levels[1] == PriceLevel(price=0.30, size=200.0)


# ---------------------------------------------------------------------------
# 10. fetch_price_history — tick parsing and ordering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_price_history_returns_sorted_ticks(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """fetch_price_history should return CLOBTick list sorted by timestamp."""
    raw_body = {
        "history": [
            {"t": 1700000300, "p": "0.62", "v": "500"},
            {"t": 1700000100, "p": "0.60", "v": "200"},
            {"t": 1700000200, "p": "0.61", "v": "300"},
        ]
    }
    mock_http_get.return_value = _make_response(200, raw_body)

    ticks = await client.fetch_price_history(
        market="0xtokBTC",
        start_ts=1700000000,
        end_ts=1700000400,
        fidelity=60,
    )

    assert len(ticks) == 3
    # Must be ascending by timestamp
    assert [t.timestamp for t in ticks] == [1700000100, 1700000200, 1700000300]
    assert all(isinstance(t, CLOBTick) for t in ticks)
    assert ticks[0].price == pytest.approx(0.60)
    assert ticks[2].price == pytest.approx(0.62)


# ---------------------------------------------------------------------------
# 11. fetch_mid_prices
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_mid_prices_returns_float_mapping(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """fetch_mid_prices should map token IDs to float mid prices."""
    token_ids = ["0xaa", "0xbb", "0xcc"]
    raw_body = {"0xaa": "0.55", "0xbb": "0.48", "0xcc": "0.72"}
    mock_http_get.return_value = _make_response(200, raw_body)

    result = await client.fetch_mid_prices(token_ids)

    assert result == {"0xaa": pytest.approx(0.55), "0xbb": pytest.approx(0.48), "0xcc": pytest.approx(0.72)}


# ---------------------------------------------------------------------------
# 12. fetch_btc_5min_markets keyword filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_btc_5min_markets_keyword_filter(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """fetch_btc_5min_markets should only return markets with 5-min keywords."""
    raw_items = [
        {
            "condition_id": "0xfive",
            "question": "Will BTC go up in the next 5 min?",
            "end_date_iso": None,
            "volume": "5000",
            "active": True,
            "tokens": [],
        },
        {
            "condition_id": "0xhour",
            "question": "Will BTC close above $80k by end of day?",
            "end_date_iso": None,
            "volume": "80000",
            "active": True,
            "tokens": [],
        },
    ]
    mock_http_get.return_value = _make_response(200, {"data": raw_items})

    markets = await client.fetch_btc_5min_markets()

    condition_ids = [m.condition_id for m in markets]
    assert "0xfive" in condition_ids
    assert "0xhour" not in condition_ids


# ---------------------------------------------------------------------------
# 13. Network error retries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_on_network_error_then_success(
    client: PolymarketCLOBClient, mock_http_get: AsyncMock
) -> None:
    """The client should retry on transient network errors and succeed."""
    success_body = {"bids": [], "asks": [{"price": "0.65", "size": "50"}]}
    success_response = _make_response(200, success_body)

    mock_http_get.side_effect = [
        httpx.ConnectError("connection refused"),
        success_response,
    ]

    with patch("heisenberg.polymarket_client.asyncio.sleep", new_callable=AsyncMock):
        book = await client.fetch_orderbook("0xnetworkerr")

    assert isinstance(book, OrderBook)
    assert book.best_ask == pytest.approx(0.65)
    assert mock_http_get.call_count == 2

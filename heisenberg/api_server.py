"""
api_server.py — FastAPI + WebSocket bridge between the SCHRODINGER bot and the dashboard.

Endpoints:
  GET  /status         → current bot_state snapshot
  GET  /signals        → last 50 tradeable signals
  GET  /markets        → markets snapshot from last cycle
  GET  /pnl            → last 50 live P&L entries
  GET  /dirac-status   → halt flag, balance, loss summary
  WS   /stream         → pushes bot_state every 1 second to all connected clients

Run with:
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import bot as bot_module
from bot import HeisenbergBot, PipelineSignal
import order_executor as _oe

load_dotenv()
logger = logging.getLogger("HEISENBERG")

# ---------------------------------------------------------------------------
# Capital fence (SCHRODINGER)
# ---------------------------------------------------------------------------

SCHRODINGER_CAPITAL_LIMIT = float(os.getenv("SCHRODINGER_CAPITAL_LIMIT", "50"))
_dirac_initial_balance: float = 0.0
_session_stop_triggered: bool = False

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="SCHRODINGER API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Bot instance
# ---------------------------------------------------------------------------

_STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "100"))
_PAPER = os.getenv("PAPER_TRADING", "true")
print(f"PAPER_TRADING env value: '{_PAPER}'")
_LIVE_MODE = _PAPER.lower() == "false"
logger.warning("SCHRODINGER MODE: %s (PAPER_TRADING='%s')",
    "LIVE TRADING" if _LIVE_MODE else "PAPER TRADING", _PAPER)
bot_instance = HeisenbergBot(bankroll=_STARTING_CAPITAL)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

bot_state: dict[str, Any] = {
    "balance": _STARTING_CAPITAL,
    "deposit": _STARTING_CAPITAL,
    "roi": 0.0,
    "win_rate": 0.0,
    "trades_hr": 0,
    "total_trades": 0,
    "sharpe": 0.0,
    "max_dd": 0.0,
    "edge": 0.0,
    "signals": [],
    "stream": [],
    "expected_edge": 0.0,
    "mode": "live" if _LIVE_MODE else "paper",
    "balance_floor_alert": False,
    "positions_open": 0,
    "status": {
        "polymarket": "ONLINE",
        "bayes": "ONLINE",
        "kelly": "ONLINE",
        "scanner": "SCAN",
        "sync": "99.8%",
    },
}

# Rolling trackers for Sharpe / drawdown
_trade_returns: list[float] = []
_peak_balance: float = _STARTING_CAPITAL
_wins: int = 0
_losses: int = 0
_cycle_start: float = time.time()
_tradeable_count: int = 0  # across all cycles for trades/hr estimate
_edges: list[float] = []
_cycle_count: int = 0
_markets_last_cycle: int = 0

# Connected WebSocket clients
_ws_clients: set[WebSocket] = set()

# Rate limiter: max 5 orders per 60-second window
_orders_this_minute: int = 0
_minute_reset: float = time.time()

# Markets with active orders: question[:60] → timestamp of last order
_active_market_orders: dict[str, float] = {}

# Markets snapshot — updated each cycle for /markets endpoint
_markets_snapshot: list[dict] = []

# P&L log — last 50 live orders
_pnl_log: list[dict] = []

# ---------------------------------------------------------------------------
# Paper trade simulation
# ---------------------------------------------------------------------------

def _simulate_trade(signal: PipelineSignal) -> None:
    """Aggressive paper trade simulation — full market payout model."""
    global _peak_balance, _wins, _losses, _tradeable_count

    balance = bot_state["balance"]

    ask = signal.spread_data.ask
    mid = signal.mid_price
    if ask <= 0 or ask >= 1 or mid <= 0 or mid >= 1:
        return

    # Position size: use Kelly if valid, fallback to net_edge-scaled 2% of bankroll
    size = signal.kelly_position_size
    if size < 0.01:
        posterior_strength = abs(signal.edge_signal.net_edge)
        size = balance * 0.02 * max(1.0, posterior_strength * 10)
    size = min(size, balance * 0.10)  # hard cap at 10% bankroll
    size = max(size, 0.50)            # minimum $0.50 per trade
    if size > balance:
        return

    # Win probability: use net_edge to shift from 50/50 baseline
    # Gives realistic 50-72% win rate range for Up/Down markets
    net_edge = signal.edge_signal.net_edge
    posterior = min(0.72, max(0.28, 0.50 + net_edge * 2.5))
    fee = 0.007 * size  # 7bps taker fee

    if random.random() < posterior:
        # YES resolves to 1.0: full payout minus fee
        pnl = size * (1.0 / ask - 1.0) - fee
        _wins += 1
        result = "WIN"
    else:
        # NO resolves: lose full stake plus fee
        pnl = -size - fee
        _losses += 1
        result = "LOSE"

    bot_state["balance"] = round(balance + pnl, 2)
    bot_state["total_trades"] = bot_state["total_trades"] + 1
    _tradeable_count += 1

    # Console log
    q_short = signal.market_question[:40]
    logger.info(
        "PAPER TRADE: market=%r size=$%.2f result=%s pnl=$%+.2f  bal=$%.2f",
        q_short, size, result, pnl, bot_state["balance"],
    )

    # Push [EXEC] stream entry
    cl = "s-tag-g" if result == "WIN" else "s-tag-r"
    bot_state["stream"].append({
        "time": _format_time(),
        "tag": "EXEC",
        "cl": cl,
        "msg": f"{q_short[:28]} | {result} size=${size:.2f} pnl=${pnl:+.2f}",
    })

    # Update peak / max drawdown
    if bot_state["balance"] > _peak_balance:
        _peak_balance = bot_state["balance"]
    drawdown = (_peak_balance - bot_state["balance"]) / _peak_balance * 100
    bot_state["max_dd"] = round(max(bot_state["max_dd"], drawdown), 2)

    # Rolling Sharpe (last 30 returns)
    _trade_returns.append(pnl / size)  # normalised return
    if len(_trade_returns) > 30:
        _trade_returns.pop(0)
    if len(_trade_returns) >= 4:
        arr = _trade_returns
        mean_r = sum(arr) / len(arr)
        std_r = math.sqrt(sum((x - mean_r) ** 2 for x in arr) / len(arr))
        bot_state["sharpe"] = round((mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0, 2)

    # Win rate
    total = _wins + _losses
    if total > 0:
        wr = _wins / total
        bot_state["win_rate"] = round(wr * 100, 1)
        # Expected edge per trade
        bot_state["expected_edge"] = round(wr * 0.08 - (1 - wr) * 0.10, 4)

    # ROI
    dep = bot_state["deposit"]
    bot_state["roi"] = round((bot_state["balance"] - dep) / dep * 100, 2)


def _format_time() -> str:
    now = datetime.now()
    return now.strftime("%H:%M:%S")


def _short_label(question: str) -> str:
    """Compress 'Bitcoin Up or Down - March 16, 7:30PM-7:45PM ET' → 'BTC 7:30-7:45PM'."""
    import re as _re
    q = question
    for full, abbr in [("Bitcoin", "BTC"), ("Ethereum", "ETH"), ("Solana", "SOL"),
                        ("Dogecoin", "DOGE"), ("HYPE", "HYPE"), ("BNB", "BNB"), ("XRP", "XRP")]:
        q = q.replace(full, abbr)
    # Extract time window e.g. "7:30PM-7:45PM"
    m = _re.search(r"(\d+:\d+[AP]M-\d+:\d+[AP]M)", q, _re.IGNORECASE)
    window = m.group(1) if m else ""
    # Get asset abbreviation (first word)
    asset = q.split()[0] if q else "?"
    return f"{asset} {window}" if window else q[:20]


def _mins_left(end_date: str | None) -> str:
    """Return '4min' or '' if end_date unavailable."""
    if not end_date:
        return ""
    try:
        from datetime import timezone as _tz
        s = end_date.rstrip("Z").split("+")[0]
        end_dt = datetime.fromisoformat(s).replace(tzinfo=_tz.utc)
        secs = (end_dt - datetime.now(_tz.utc)).total_seconds()
        if secs <= 0:
            return "0min"
        return f"{int(secs/60)}min"
    except Exception:
        return ""


def _signal_to_stream(signal: PipelineSignal, tag: str, cl: str, msg: str) -> dict:
    return {"time": _format_time(), "tag": tag, "cl": cl, "msg": msg}


# ---------------------------------------------------------------------------
# Cycle callback (called by bot after each cycle)
# ---------------------------------------------------------------------------

async def _on_cycle_complete(signals: list[PipelineSignal]) -> None:
    global _tradeable_count, _cycle_count, _markets_last_cycle, _markets_snapshot
    _cycle_count += 1
    _markets_last_cycle = len(signals)

    if signals:
        z_vals = [s.edge_signal.z_score for s in signals]
        net_vals = [s.edge_signal.net_edge for s in signals]
        tradeable_n = sum(1 for s in signals if s.edge_signal.is_tradeable)
        logger.info(
            "Cycle %d — %d tokens | z=[%.3f..%.3f] net=[%.4f..%.4f] | %d tradeable",
            _cycle_count, len(signals),
            min(z_vals), max(z_vals),
            min(net_vals), max(net_vals),
            tradeable_n,
        )

    tradeable = [s for s in signals if s.edge_signal.is_tradeable]

    # Update markets snapshot for /markets endpoint
    _markets_snapshot = [
        {
            "question": s.market_question[:80],
            "token_id": s.token_id,
            "mid_price": round(s.mid_price, 4),
            "z": round(s.edge_signal.z_score, 3),
            "tradeable": s.edge_signal.is_tradeable,
        }
        for s in signals
    ]

    # Collect edges
    for s in signals:
        _edges.append(abs(s.edge_signal.net_edge))
    if len(_edges) > 100:
        _edges[:] = _edges[-100:]
    if _edges:
        bot_state["edge"] = round(sum(_edges) / len(_edges) * 100, 2)

    # Trades/hr estimate (no cap — reflects actual signal rate)
    elapsed_hrs = max((time.time() - _cycle_start) / 3600, 1 / 3600)
    bot_state["trades_hr"] = int(_tradeable_count / elapsed_hrs)

    # Execute trades — live or paper
    if _LIVE_MODE:
        await _cancel_then_place(tradeable)
    else:
        for s in tradeable:
            _simulate_trade(s)

    # Build stream entries
    new_entries: list[dict] = []
    for s in signals[:8]:  # cap to avoid flood
        label = _short_label(s.market_question)
        left = _mins_left(s.end_date)
        time_str = f" | {left} left" if left else ""
        if s.edge_signal.is_tradeable:
            msg = f"{label}{time_str} | z={s.edge_signal.z_score:+.3f} size=${s.kelly_position_size:.2f}"
            new_entries.append({"time": _format_time(), "tag": "SIGNAL", "cl": "s-tag-g", "msg": msg})
        else:
            msg = f"{label}{time_str} | z={s.edge_signal.z_score:+.3f} net={s.edge_signal.net_edge:+.4f}"
            new_entries.append({"time": _format_time(), "tag": "SKIP", "cl": "s-tag-b", "msg": msg})

    # Append SCAN heartbeat
    new_entries.append({
        "time": _format_time(),
        "tag": "SCAN",
        "cl": "s-tag-y",
        "msg": f"{len(signals)} tokens · {len(tradeable)} signals",
    })

    stream = bot_state["stream"]
    stream.extend(new_entries)
    bot_state["stream"] = stream[-100:]

    # Append tradeable signals to signals list
    for s in tradeable:
        bot_state["signals"].append({
            "token_id": s.token_id,
            "question": s.market_question[:60],
            "mid": s.mid_price,
            "z": s.edge_signal.z_score,
            "ev": s.edge_signal.expected_value,
            "edge": s.edge_signal.net_edge,
            "size": s.kelly_position_size,
            "time": _format_time(),
        })
    bot_state["signals"] = bot_state["signals"][-50:]


async def _cancel_then_place(signals: list[PipelineSignal]) -> None:
    global _orders_this_minute, _minute_reset
    # SCHRODINGER HALT: block all trades if stop-loss triggered
    if _session_stop_triggered:
        logger.info("SCHRODINGER HALT active — skipping all trades")
        return
    now = time.time()
    if now - _minute_reset > 60:
        _orders_this_minute = 0
        _minute_reset = now
    if _orders_this_minute >= 20:
        logger.info("Rate limit reached — skipping")
        return
    MAX_CONCURRENT = 3
    if bot_state.get("positions_open", 0) >= MAX_CONCURRENT:
        logger.info("Max concurrent positions reached — skipping")
        return
    for s in signals:
        await _place_live_order(s)


async def _place_live_order(signal: PipelineSignal) -> None:
    global _orders_this_minute
    # SCHRODINGER deduplication — clean 300s window per market
    market_key = signal.market_question[:60]
    now = time.time()
    if market_key in _active_market_orders:
        elapsed = now - _active_market_orders[market_key]
        if elapsed < 300:
            logger.info("SCHRODINGER: %s blocked %.0fs more",
                market_key[:30], 300 - elapsed)
            return
    _active_market_orders[market_key] = now

    direction = "BUY" if signal.edge_signal.net_edge > 0 else "SELL"
    offset = -0.005 if direction == "BUY" else +0.005
    price = round(max(0.01, min(0.99, signal.mid_price + offset)), 3)
    dollar_size = max(signal.kelly_position_size, 0.0)
    shares = dollar_size / price if price > 0 else 0.0
    shares = max(1.0, round(shares, 2))
    _orders_this_minute += 1
    result = await _oe.place_order(signal.token_id, direction, price, shares)
    label = _short_label(signal.market_question)
    if result:
        bot_state["total_trades"] += 1
        bot_state["positions_open"] = bot_state.get("positions_open", 0) + 1
        bot_state["stream"].append({
            "time": _format_time(),
            "tag": "ORDER",
            "cl": "s-tag-g",
            "msg": f"{label} | {direction} ${dollar_size:.2f} @ {price:.3f}",
        })
        logger.info("ORDER PLACED: %s %s $%.2f @ %.3f",
            direction, label, dollar_size, price)
        _pnl_log.append({
            "time": _format_time(),
            "market": signal.market_question[:60],
            "direction": direction,
            "price": price,
            "shares": shares,
            "dollar_size": dollar_size,
        })
    else:
        _active_market_orders.pop(market_key, None)
        _orders_this_minute = max(0, _orders_this_minute - 1)
        bot_state["stream"].append({
            "time": _format_time(),
            "tag": "ERR",
            "cl": "s-tag-r",
            "msg": f"Order failed — {label}",
        })


# Register callback with bot module
bot_module.on_cycle_complete = _on_cycle_complete

# ---------------------------------------------------------------------------
# WebSocket broadcaster
# ---------------------------------------------------------------------------

async def _sync_live_balance() -> None:
    """Fetch real CLOB balance and open positions, update bot_state."""
    global _dirac_initial_balance, _session_stop_triggered
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
        client = await _oe._get_client()
        if not client:
            return
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=2)
        data = await asyncio.to_thread(client.get_balance_allowance, params)
        raw = int(data.get("balance", "0"))
        usdc_balance = raw / 1_000_000  # 6 decimals
        bot_state["balance"] = round(usdc_balance, 2)
        dep = bot_state["deposit"]
        bot_state["roi"] = round((usdc_balance - dep) / dep * 100, 2) if dep > 0 else 0.0

        # SCHRODINGER stop-loss: halt if loss >= capital limit or >= 20% of session start
        if _dirac_initial_balance == 0.0:
            _dirac_initial_balance = bot_state["balance"]
        total_loss = _dirac_initial_balance - bot_state["balance"]
        session_loss_pct = total_loss / _dirac_initial_balance if _dirac_initial_balance > 0 else 0
        if total_loss >= SCHRODINGER_CAPITAL_LIMIT or session_loss_pct >= 0.20:
            _session_stop_triggered = True
            logger.warning("SCHRODINGER HALT: loss $%.2f (%.0f%%) — trading stopped",
                total_loss, session_loss_pct * 100)
    except Exception as e:
        logger.debug("balance sync failed: %s", e)

    try:
        client = await _oe._get_client()
        if client:
            orders = await asyncio.to_thread(client.get_orders)
            if orders:
                logger.info("ORDER STATUSES: %s",
                    list(set(o.get("status") for o in orders)))
                open_orders = [o for o in orders
                    if float(o.get("size_matched", 0)) > 0
                    and float(o.get("size_matched", 0)) < float(o.get("original_size", o.get("size", 1)))]
                bot_state["positions_open"] = len(open_orders)
            else:
                bot_state["positions_open"] = 0
    except Exception as e:
        logger.debug("positions sync failed: %s", e)
        bot_state["positions_open"] = 0


async def _broadcast_loop() -> None:
    """Push bot_state snapshot to all WS clients every 1s; expire stale orders every 30s."""
    _expire_tick = 0
    while True:
        await asyncio.sleep(1)
        _expire_tick += 1
        if _LIVE_MODE and _expire_tick % 10 == 0:
            await _sync_live_balance()

        if not _ws_clients:
            continue
        payload = json.dumps(bot_state)
        dead: set[WebSocket] = set()
        for ws in list(_ws_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        _ws_clients.difference_update(dead)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup() -> None:
    if _LIVE_MODE:
        logger.warning("*** LIVE TRADING ENABLED — REAL USDC ***")
    else:
        logger.info("PAPER TRADING MODE — no real orders will be placed.")
    asyncio.create_task(_oe._run_startup_allowance())
    asyncio.create_task(bot_instance.run())
    asyncio.create_task(_broadcast_loop())
    logger.info("SCHRODINGER API started. Bot running. Broadcaster running.")


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cycles": _cycle_count,
        "markets": _markets_last_cycle,
        "balance": bot_state["balance"],
        "uptime_s": round(time.time() - _cycle_start),
    }


@app.get("/status")
async def get_status():
    return bot_state


@app.get("/signals")
async def get_signals():
    return bot_state["signals"][-50:]


@app.get("/markets")
async def get_markets():
    return {
        "cycle": _cycle_count,
        "count": len(_markets_snapshot),
        "markets": _markets_snapshot,
    }


@app.get("/search-test")
async def search_test():
    """Debug endpoint: runs the Up/Down market search and returns raw results."""
    import httpx as _httpx
    GAMMA = "https://gamma-api.polymarket.com"
    async with _httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{GAMMA}/events", params={
            "closed": "false", "limit": 200, "tag_slug": "crypto",
            "order": "liquidity", "ascending": "false",
        })
        events = r.json() if isinstance(r.json(), list) else r.json().get("data", [])
    updown = [
        {
            "title": e.get("title"),
            "liquidity": e.get("liquidity"),
            "endDate": e.get("endDate"),
            "market_count": len(e.get("markets", [])),
            "accepting": any(m.get("acceptingOrders") for m in e.get("markets", [])),
            "tokens": sum(
                len(m.get("clobTokenIds") or []) for m in e.get("markets", [])
            ),
        }
        for e in events
        if "up or down" in (e.get("title") or "").lower()
    ]
    return {
        "total_crypto_events": len(events),
        "up_down_markets": len(updown),
        "markets": updown,
    }


@app.get("/pnl")
async def get_pnl():
    return _pnl_log[-50:]


@app.get("/dirac-status")
async def dirac_status():
    return {
        "halt": _session_stop_triggered,
        "initial_balance": _dirac_initial_balance,
        "current_balance": bot_state["balance"],
        "total_loss": _dirac_initial_balance - bot_state["balance"],
        "capital_limit": SCHRODINGER_CAPITAL_LIMIT,
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.add(ws)
    try:
        # Send current state immediately on connect
        await ws.send_text(json.dumps(bot_state))
        # Keep alive — bot broadcaster handles sends
        while True:
            await asyncio.sleep(30)
            await ws.send_text('{"ping":1}')
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _ws_clients.discard(ws)

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  ██╗  ██╗███████╗██╗███████╗███████╗███╗   ██╗██████╗ ███████╗██████╗  ██████╗  ║
║  ██║  ██║██╔════╝██║██╔════╝██╔════╝████╗  ██║██╔══██╗██╔════╝██╔══██╗██╔════╝  ║
║  ███████║█████╗  ██║███████╗█████╗  ██╔██╗ ██║██████╔╝█████╗  ██████╔╝██║  ███╗ ║
║  ██╔══██║██╔══╝  ██║╚════██║██╔══╝  ██║╚██╗██║██╔══██╗██╔══╝  ██╔══██╗██║   ██║ ║
║  ██║  ██║███████╗██║███████║███████╗██║ ╚████║██████╔╝███████╗██║  ██║╚██████╔╝ ║
║  ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝  ║
║                                                                              ║
║  "The more precisely the position is determined, the less precisely the     ║
║   momentum is known in this instant, and vice versa."  — W. Heisenberg      ║
║                                                                              ║
║  Polymarket 5-Minute BTC Arbitrage Bot                                      ║
║  READ-ONLY MODE  |  NO LIVE TRADING  |  NO WALLET                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from polymarket_client import PolymarketCLOBClient, OrderBook, MarketInfo

# ---------------------------------------------------------------------------
# Cycle callback — set by api_server.py to receive live updates
# ---------------------------------------------------------------------------
on_cycle_complete: Optional[Callable] = None
from bayesian_model import BayesianModel, MarketFeatures, compute_ewma_vol
from edge_filter import EdgeFilter, SpreadData, EdgeSignal
from kelly_sizing import KellySizer, KellyInput
from stoikov_quoting import StoikovQuoter, StoikovParams


def detect_trend(history: list[float]) -> str:
    if len(history) < 10:
        return 'NEUTRAL'
    recent = history[-5:]
    older = history[-10:-5]
    avg_recent = sum(recent) / len(recent)
    avg_older = sum(older) / len(older)
    diff = avg_recent - avg_older
    if diff > 0.03:
        return 'UP'
    elif diff < -0.03:
        return 'DOWN'
    return 'NEUTRAL'


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HEISENBERG")

# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 2        # Aggressive: scan every 2 seconds
MAX_MARKETS_PER_CYCLE = 0        # 0 = no cap, scan everything
BANKROLL = float(os.getenv("STARTING_CAPITAL", "100"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.20"))  # Conservative fractional Kelly
MIN_EDGE_BPS = 10                # Low threshold — z-score is the gate
MIN_Z_SCORE = float(os.getenv("MIN_Z_SCORE", "2.0"))
MAX_SPREAD_BPS = 800             # Relaxed spread tolerance

# Market health filters — applied per token after book fetch
MID_PRICE_MIN = 0.05             # Skip near-resolved NO (< 5%)
MID_PRICE_MAX = 0.95             # Skip near-resolved YES (> 95%)
MAX_SPREAD_FILTER = 0.10         # Skip illiquid books (spread > 10%)


@dataclass
class PipelineSignal:
    """Full signal record produced for one market token per cycle."""

    token_id: str
    market_question: str
    mid_price: float
    spread_data: SpreadData
    edge_signal: EdgeSignal
    kelly_position_size: float
    reservation_price: float
    bid_quote: float
    ask_quote: float
    end_date: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        direction = "BUY" if self.edge_signal.net_edge > 0 else "SELL"
        return (
            f"[{'SIGNAL' if self.edge_signal.is_tradeable else 'SKIP  '}] "
            f"{self.market_question[:50]:<50} | "
            f"mid={self.mid_price:.3f} | "
            f"z={self.edge_signal.z_score:+.2f} | "
            f"ev={self.edge_signal.expected_value:+.4f} | "
            f"edge={self.edge_signal.net_edge:+.4f} | "
            f"size=${self.kelly_position_size:.2f} | "
            f"quotes=[{self.bid_quote:.3f}, {self.ask_quote:.3f}] | "
            f"{direction}"
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class HeisenbergBot:
    """
    Main orchestrator connecting all HEISENBERG modules in a single pipeline:

        PolymarketClient
            → BayesianModel  (posterior probability of mispricing)
            → EdgeFilter     (z-score, EV, net edge signal)
            → KellySizer     (fractional Kelly position sizing)
            → StoikovQuoter  (reservation price + optimal quotes)
            → PipelineSignal (output)
    """

    def __init__(self, bankroll: float = BANKROLL) -> None:
        self.bankroll = bankroll
        self.client = PolymarketCLOBClient()
        self.bayesian = BayesianModel(prior=0.5)
        self.edge_filter = EdgeFilter(
            min_edge_bps=MIN_EDGE_BPS,
            min_z_score=MIN_Z_SCORE,
            max_spread_bps=MAX_SPREAD_BPS,
        )
        self.kelly = KellySizer(kelly_fraction=KELLY_FRACTION, max_position_pct=0.10)
        self.stoikov = StoikovQuoter(StoikovParams(gamma=0.1, sigma=0.02, T=300.0))

        # Rolling price history per token for z-score computation
        self._price_history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Per-token signal computation
    # ------------------------------------------------------------------

    async def _process_token(
        self, token_id: str, question: str, elapsed_t: float, end_date: Optional[str] = None
    ) -> Optional[PipelineSignal]:
        """Run full pipeline for one token. Returns None on data failure."""
        # 1. Fetch live orderbook
        try:
            book: OrderBook = await self.client.fetch_orderbook(token_id)
        except Exception as exc:
            logger.warning("fetch_orderbook failed for %s: %s", token_id, exc)
            return None

        if book.best_bid is None or book.best_ask is None:
            logger.debug("Empty book for token %s, skipping.", token_id)
            return None

        mid = book.mid_price
        bid = book.best_bid
        ask = book.best_ask
        spread = ask - bid

        # Filter near-resolved markets and illiquid books
        if mid < MID_PRICE_MIN or mid > MID_PRICE_MAX:
            logger.debug(
                "Skipping near-resolved token %s mid=%.3f (out of [%.2f, %.2f])",
                token_id[:12], mid, MID_PRICE_MIN, MID_PRICE_MAX,
            )
            return None
        if spread > MAX_SPREAD_FILTER:
            logger.debug(
                "Skipping illiquid token %s spread=%.3f > %.2f",
                token_id[:12], spread, MAX_SPREAD_FILTER,
            )
            return None

        # 2. Update rolling price history
        history = self._price_history.setdefault(token_id, [])
        history.append(mid)
        if len(history) > 200:
            history.pop(0)

        # 3. Compute market features for Bayesian model
        bid_vol = sum(pl.size for pl in book.bids)
        ask_vol = sum(pl.size for pl in book.asks)
        total_vol = bid_vol + ask_vol
        book_imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0

        vol = compute_ewma_vol(history, span=20)
        # spot_delta: deviation of mid from neutral 0.5 (fair coin prior)
        spot_delta = (mid - 0.5) / max(vol, 1e-6) if vol > 0 else 0.0

        features = MarketFeatures(
            spot_delta=spot_delta,
            volatility=vol,
            book_imbalance=book_imbalance,
            spread=ask - bid,
            mid_price=mid,
        )

        # 4. Bayesian posterior
        signal = self.bayesian.compute_signal(features)
        posterior = signal.posterior_prob

        # 5. Spread data
        spread_data = self.edge_filter.compute_spread(bid, ask)

        # 6. Z-score + EV + net edge
        z_score = self.edge_filter.compute_z_score(mid, history[:-1], window=20)
        logger.debug(
            "token=%s mid=%.3f z=%.3f ev=%.4f hist_len=%d",
            token_id[:12], mid, z_score,
            0.0,  # ev computed below
            len(history),
        )
        ev = self.edge_filter.compute_ev(
            prob=posterior,
            odds_yes=1.0 / ask if ask > 0 else 0.0,
            odds_no=1.0 / (1.0 - bid) if bid < 1.0 else 0.0,
            fee_bps=20,
        )
        edge_signal = self.edge_filter.filter(spread_data, z_score, ev, posterior)
        logger.info(
            "  %-12s mid=%.3f spread=%.3f z=%+.3f ev=%+.4f net=%+.4f %s",
            token_id[:12], mid, spread, z_score, ev,
            edge_signal.net_edge,
            "SIGNAL" if edge_signal.is_tradeable else "skip",
        )

        # 7. Kelly position sizing
        odds_win = 1.0 / ask if ask > 0 else 1.0
        kelly_input = KellyInput(
            prob_win=posterior,
            odds_win=odds_win,
            odds_lose=1.0 / (1.0 - bid) if bid < 1.0 else 1.0,
            bankroll=self.bankroll,
            kelly_fraction=KELLY_FRACTION,
        )
        kelly_result = self.kelly.compute_kelly(kelly_input)
        logger.debug(
            "KELLY DEBUG: prob=%.3f odds_win=%.3f b=%.3f full_kelly=%.4f frac_kelly=%.4f size=$%.2f",
            posterior, odds_win, odds_win - 1,
            kelly_result.full_kelly, kelly_result.fractional_kelly, kelly_result.position_size,
        )

        # 8. Stoikov reservation price + optimal quotes
        inventory = 0.0  # neutral inventory (no positions held yet)
        quotes = self.stoikov.compute_quotes(mid, inventory, elapsed_t)

        trend = detect_trend(history)
        direction = "BUY" if edge_signal.net_edge > 0 else "SELL"
        if trend == 'UP' and direction == "SELL":
            logger.info("SCHRODINGER trend filter: skipping SELL in UP trend")
            return None
        if trend == 'DOWN' and direction == "BUY":
            logger.info("SCHRODINGER trend filter: skipping BUY in DOWN trend")
            return None

        return PipelineSignal(
            token_id=token_id,
            market_question=question,
            mid_price=mid,
            spread_data=spread_data,
            edge_signal=edge_signal,
            kelly_position_size=kelly_result.position_size,
            reservation_price=quotes.reservation_price,
            bid_quote=quotes.bid_quote,
            ask_quote=quotes.ask_quote,
            end_date=end_date,
        )

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def _fetch_active_btc_markets(self) -> list[MarketInfo]:
        """Fetch short-horizon crypto markets from Polymarket, sorted by volume."""
        try:
            markets = await self.client.fetch_short_horizon_markets()
            if MAX_MARKETS_PER_CYCLE > 0:
                return markets[:MAX_MARKETS_PER_CYCLE]
            return markets
        except Exception as exc:
            logger.warning("Market discovery failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Single scan cycle
    # ------------------------------------------------------------------

    async def run_cycle(self, cycle_num: int) -> list[PipelineSignal]:
        """Execute one full scan cycle across all active BTC markets."""
        logger.info("── Cycle %d ──────────────────────────────────", cycle_num)
        markets = await self._fetch_active_btc_markets()

        if not markets:
            logger.warning("No BTC markets found this cycle.")
            return []

        elapsed_t = (cycle_num * POLL_INTERVAL_SECONDS) % 300  # within 5-min window

        tasks = []
        for market in markets:
            for token in market.tokens:
                token_id = token.get("token_id", "") if isinstance(token, dict) else str(token)
                if token_id:
                    tasks.append(self._process_token(token_id, market.question, elapsed_t, market.end_date))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: list[PipelineSignal] = []
        for r in results:
            if isinstance(r, PipelineSignal):
                signals.append(r)
                logger.info(r.summary())
            elif isinstance(r, Exception):
                logger.debug("Token processing error: %s", r)

        # Deduplicate: for each market event keep only the one token with the
        # highest absolute net_edge.  Two tokens share the same market_question
        # (YES and NO sides of the same Up/Down window), so trading both would
        # mean taking both sides — guaranteed to cancel out PnL.
        best: dict[str, PipelineSignal] = {}
        for s in signals:
            key = s.market_question  # same for both YES/NO tokens
            if key not in best or abs(s.edge_signal.net_edge) > abs(best[key].edge_signal.net_edge):
                best[key] = s
        if len(signals) != len(best):
            logger.info(
                "Dedup: %d tokens → %d (dropped %d same-market duplicates)",
                len(signals), len(best), len(signals) - len(best),
            )
        signals = list(best.values())

        tradeable = [s for s in signals if s.edge_signal.is_tradeable]
        logger.info(
            "Cycle %d complete — %d tokens scanned, %d tradeable signals",
            cycle_num, len(signals), len(tradeable),
        )

        if on_cycle_complete is not None:
            logger.info("SCHRODINGER: calling on_cycle_complete (tradeable=%d)", len(tradeable))
            try:
                await on_cycle_complete(signals)
            except Exception as cb_exc:
                logger.error("on_cycle_complete ERROR: %s", cb_exc, exc_info=True)
        else:
            logger.warning("SCHRODINGER: on_cycle_complete is None — api_server not connected")

        return signals

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, max_cycles: int = 0) -> None:
        """
        Main event loop. Runs indefinitely (max_cycles=0) or for N cycles.
        READ-ONLY: logs signals only, no orders placed.
        """
        logger.info("SCHRODINGER bot starting — LIVE TRADING MODE")
        logger.info("Bankroll: $%.2f | Kelly: %.0f%% | Poll: %ds",
                    self.bankroll, KELLY_FRACTION * 100, POLL_INTERVAL_SECONDS)

        cycle = 0
        try:
            while True:
                cycle += 1
                await self.run_cycle(cycle)

                if max_cycles and cycle >= max_cycles:
                    logger.info("Reached max_cycles=%d, shutting down.", max_cycles)
                    break

                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            logger.info("Bot cancelled — shutting down cleanly.")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — shutting down.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    bot = HeisenbergBot(bankroll=BANKROLL)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())

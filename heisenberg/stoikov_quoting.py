"""
stoikov_quoting.py — Avellaneda-Stoikov market-making model adapted for
binary prediction markets (HEISENBERG / Polymarket arbitrage bot).

Python 3.11, numpy only.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class StoikovParams:
    """Parameters for the Avellaneda-Stoikov model.

    Attributes
    ----------
    gamma:  Risk-aversion coefficient. Higher values produce wider spreads and
            stronger inventory skew corrections. Default 0.1.
    sigma:  Volatility of the mid-price (per second, in probability units).
            Default 0.02.
    T:      Time horizon in seconds. Default 300 (5-minute markets).
    dt:     Time step in seconds. Default 1.0.
    kappa:  Order-arrival intensity (market order rate). Default 1.5.
    """

    gamma: float = 0.1
    sigma: float = 0.02
    T: float = 300.0
    dt: float = 1.0
    kappa: float = 1.5


@dataclass
class QuoteResult:
    """Output of a single quoting calculation.

    Attributes
    ----------
    reservation_price: Inventory-adjusted fair value.
    bid_quote:         Price at which we post a buy order.
    ask_quote:         Price at which we post a sell order.
    spread:            ask_quote - bid_quote (always >= 0.01).
    inventory_skew:    abs(inventory) — magnitude of the current position skew.
    """

    reservation_price: float
    bid_quote: float
    ask_quote: float
    spread: float
    inventory_skew: float


class StoikovQuoter:
    """Avellaneda-Stoikov quoter adapted for binary prediction markets.

    Quotes are always clamped to [0.01, 0.99] to respect the hard
    probability bounds of a binary (YES/NO) market.

    Parameters
    ----------
    params : StoikovParams
        Model configuration.
    """

    MIN_PRICE: float = 0.01
    MAX_PRICE: float = 0.99
    MIN_SPREAD: float = 0.01

    def __init__(self, params: StoikovParams) -> None:
        self.params = params

    # ------------------------------------------------------------------
    # Core model equations
    # ------------------------------------------------------------------

    def compute_reservation_price(
        self, mid: float, inventory: float, t: float
    ) -> float:
        """Compute the inventory-adjusted reservation price.

        Formula
        -------
        r = mid - inventory * gamma * sigma^2 * (T - t)

        A positive inventory (long position) pushes the reservation price
        below the mid, incentivising the bot to sell.  A negative inventory
        (short position) pushes it above the mid, incentivising buys.

        Parameters
        ----------
        mid:       Current mid-price in (0, 1).
        inventory: Signed position in [-1, 1].  Positive = net long.
        t:         Elapsed time in seconds since the quoting session began.

        Returns
        -------
        float
            Reservation price, clamped to [MIN_PRICE, MAX_PRICE].
        """
        p = self.params
        time_remaining = p.T - t
        r = mid - inventory * p.gamma * (p.sigma ** 2) * time_remaining
        return float(np.clip(r, self.MIN_PRICE, self.MAX_PRICE))

    def compute_optimal_spread(self, t: float) -> float:
        """Compute the theoretically optimal bid-ask spread.

        Formula
        -------
        spread = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)

        The first term captures inventory risk (widens when time is plentiful).
        The second term captures the adverse-selection / arrival-rate component.

        Parameters
        ----------
        t: Elapsed time in seconds.

        Returns
        -------
        float
            Optimal spread, floored at MIN_SPREAD.
        """
        p = self.params
        time_remaining = max(p.T - t, 0.0)
        inventory_risk_term = p.gamma * (p.sigma ** 2) * time_remaining
        arrival_rate_term = (2.0 / p.gamma) * np.log(1.0 + p.gamma / p.kappa)
        spread = inventory_risk_term + arrival_rate_term
        return float(max(spread, self.MIN_SPREAD))

    def compute_quotes(
        self, mid: float, inventory: float, t: float
    ) -> QuoteResult:
        """Compute bid/ask quotes using the Avellaneda-Stoikov model.

        Steps
        -----
        1. Compute reservation price.
        2. Compute optimal spread.
        3. Set bid = r - spread/2, ask = r + spread/2.
        4. Clamp both to [MIN_PRICE, MAX_PRICE].
        5. Pass through binary-market safety adjustment.

        Parameters
        ----------
        mid:       Current mid-price in (0, 1).
        inventory: Signed inventory in [-1, 1].
        t:         Elapsed time in seconds.

        Returns
        -------
        QuoteResult
        """
        r = self.compute_reservation_price(mid, inventory, t)
        spread = self.compute_optimal_spread(t)

        bid_raw = r - spread / 2.0
        ask_raw = r + spread / 2.0

        bid = float(np.clip(bid_raw, self.MIN_PRICE, self.MAX_PRICE))
        ask = float(np.clip(ask_raw, self.MIN_PRICE, self.MAX_PRICE))

        inventory_skew = abs(inventory)

        raw_result = QuoteResult(
            reservation_price=r,
            bid_quote=bid,
            ask_quote=ask,
            spread=ask - bid,
            inventory_skew=inventory_skew,
        )
        return self.adjust_for_binary_market(raw_result)

    # ------------------------------------------------------------------
    # Binary-market safety layer
    # ------------------------------------------------------------------

    def adjust_for_binary_market(self, quotes: QuoteResult) -> QuoteResult:
        """Enforce binary-market constraints on a QuoteResult.

        Rules
        -----
        * Both bid and ask must be in (0, 1); we use [MIN_PRICE, MAX_PRICE].
        * bid < ask must hold.
        * Minimum spread of MIN_SPREAD (0.01) must be maintained.

        If clamping causes bid >= ask, we symmetrically widen around the
        reservation price until the minimum spread is satisfied.

        Parameters
        ----------
        quotes : QuoteResult
            Possibly-invalid quotes to sanitise.

        Returns
        -------
        QuoteResult
            Sanitised quotes guaranteed to satisfy all constraints.
        """
        bid = float(np.clip(quotes.bid_quote, self.MIN_PRICE, self.MAX_PRICE))
        ask = float(np.clip(quotes.ask_quote, self.MIN_PRICE, self.MAX_PRICE))
        r = quotes.reservation_price

        # Enforce minimum spread
        if ask - bid < self.MIN_SPREAD:
            half = self.MIN_SPREAD / 2.0
            bid = float(np.clip(r - half, self.MIN_PRICE, self.MAX_PRICE))
            ask = float(np.clip(r + half, self.MIN_PRICE, self.MAX_PRICE))

            # Edge case: both sides got pinned to the same boundary.
            # Shift ask up or bid down to guarantee the minimum gap.
            if ask - bid < self.MIN_SPREAD:
                if ask < self.MAX_PRICE:
                    ask = float(
                        np.clip(bid + self.MIN_SPREAD, self.MIN_PRICE, self.MAX_PRICE)
                    )
                else:
                    bid = float(
                        np.clip(ask - self.MIN_SPREAD, self.MIN_PRICE, self.MAX_PRICE)
                    )

        actual_spread = ask - bid

        return QuoteResult(
            reservation_price=quotes.reservation_price,
            bid_quote=bid,
            ask_quote=ask,
            spread=actual_spread,
            inventory_skew=quotes.inventory_skew,
        )

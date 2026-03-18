"""
HEISENBERG — Bayesian Probability Model
Project: Polymarket Arbitrage Bot
Agent: BAYESIAN_AGENT

Implements P(D|H) = f(spot_delta, vol, book_imbalance) using pure numpy/scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketFeatures:
    """
    Observed market features used as evidence in the Bayesian update.

    Attributes
    ----------
    spot_delta      : Normalised price deviation from theoretical fair value.
                      Positive  => market price > fair value (potentially overpriced).
                      Negative  => market price < fair value (potentially underpriced).
    volatility      : Recent realised volatility (EWMA-based, annualised or per-period).
    book_imbalance  : (bid_vol - ask_vol) / (bid_vol + ask_vol), range [-1, 1].
                      Positive  => buy-side pressure.
                      Negative  => sell-side pressure.
    spread          : Best-ask minus best-bid (absolute, in probability points).
    mid_price       : Mid-point of best bid/ask, range [0, 1] for binary markets.
    """

    spot_delta: float
    volatility: float
    book_imbalance: float
    spread: float
    mid_price: float

    def __post_init__(self) -> None:
        # Clip book_imbalance to its valid domain
        object.__setattr__(
            self,
            "book_imbalance",
            float(np.clip(self.book_imbalance, -1.0, 1.0)),
        )
        # Ensure spread and volatility are non-negative
        object.__setattr__(self, "spread", max(0.0, float(self.spread)))
        object.__setattr__(self, "volatility", max(0.0, float(self.volatility)))


@dataclass
class BayesianSignal:
    """
    Output of a single Bayesian inference step.

    Attributes
    ----------
    posterior_prob  : P(H|D) — probability market is mispriced given evidence.
    likelihood      : P(D|H) — probability of observing evidence if mispriced.
    prior           : P(H)   — prior probability of mispricing (before update).
    log_odds        : log( posterior / (1 - posterior) ) — signed signal in log space.
    signal_strength : Normalised [0, 1] measure; 0.5 = no signal, 1.0 = max long,
                      0.0 = max short.
    """

    posterior_prob: float
    likelihood: float
    prior: float
    log_odds: float
    signal_strength: float


# ---------------------------------------------------------------------------
# EWMA Volatility Helper
# ---------------------------------------------------------------------------

def compute_ewma_vol(prices: list[float], span: int = 20) -> float:
    """
    Compute the Exponentially Weighted Moving Average (EWMA) realised volatility
    from a price series.

    Parameters
    ----------
    prices : list[float]
        Ordered price observations (oldest first).  Requires at least 2 points.
    span   : int
        EWMA span parameter.  Decay factor α = 2 / (span + 1).

    Returns
    -------
    float
        Annualised-equivalent EWMA volatility (std of log-returns, EWMA-weighted).
        Returns 0.0 for fewer than 2 prices or a perfectly flat series.
    """
    if len(prices) < 2:
        return 0.0

    arr = np.array(prices, dtype=np.float64)

    # Guard against non-positive prices before taking log
    if np.any(arr <= 0.0):
        # Fall back to simple returns if prices can be zero/negative
        returns = np.diff(arr)
    else:
        returns = np.diff(np.log(arr))

    if len(returns) == 0:
        return 0.0

    alpha = 2.0 / (span + 1)

    # EWMA variance (initialised to first squared return)
    ewma_var = float(returns[0] ** 2)
    for r in returns[1:]:
        ewma_var = alpha * float(r ** 2) + (1.0 - alpha) * ewma_var

    vol = math.sqrt(max(ewma_var, 0.0))
    return vol


# ---------------------------------------------------------------------------
# Sigmoid utility
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid: 1 / (1 + exp(-x)).
    Clips input to [-500, 500] to prevent overflow.
    """
    x_clipped = float(np.clip(x, -500.0, 500.0))
    return 1.0 / (1.0 + math.exp(-x_clipped))


# ---------------------------------------------------------------------------
# BayesianModel
# ---------------------------------------------------------------------------

class BayesianModel:
    """
    Bayesian mispricing detector for binary prediction markets.

    The model treats "is this market mispriced?" as hypothesis H and uses
    observed market features as evidence D.

    Inference
    ---------
        P(H|D) = P(D|H) * P(H) / P(D)
        P(D)   = P(D|H) * P(H) + P(D|~H) * (1 - P(H))
        P(D|~H) = 1 - P(D|H)          (complement assumption)

    Likelihood
    ----------
        P(D|H) = sigmoid(w1 * spot_delta + w2 * book_imbalance - w3 * vol)

    Parameters
    ----------
    prior : float
        Prior probability that a given market is mispriced.  Default 0.5
        (maximum uncertainty / uninformative prior).
    w1, w2, w3 : float
        Weights for spot_delta, book_imbalance, and volatility respectively.
    """

    _DEFAULT_PRIOR: float = 0.5
    _DEFAULT_W1: float = 2.5   # spot_delta weight
    _DEFAULT_W2: float = 1.8   # book_imbalance weight
    _DEFAULT_W3: float = 0.9   # volatility penalty

    def __init__(
        self,
        prior: float = 0.5,
        w1: float = _DEFAULT_W1,
        w2: float = _DEFAULT_W2,
        w3: float = _DEFAULT_W3,
    ) -> None:
        if not (0.0 < prior < 1.0):
            raise ValueError(f"prior must be in (0, 1), got {prior!r}")
        self._initial_prior: float = prior
        self.prior: float = prior
        self.w1: float = w1
        self.w2: float = w2
        self.w3: float = w3

    # ------------------------------------------------------------------
    # Core inference methods
    # ------------------------------------------------------------------

    def compute_likelihood(self, features: MarketFeatures) -> float:
        """
        Compute P(D|H) — likelihood of observing evidence given mispricing.

        P(D|H) = sigmoid(w1 * spot_delta + w2 * book_imbalance - w3 * vol)

        Inputs are soft-clipped before the sigmoid to ensure numerical stability.

        Parameters
        ----------
        features : MarketFeatures

        Returns
        -------
        float in (0, 1)
        """
        # Soft-clip individual features to prevent degenerate linear combinations
        spot_delta = float(np.clip(features.spot_delta, -10.0, 10.0))
        book_imbalance = float(np.clip(features.book_imbalance, -1.0, 1.0))
        vol = float(np.clip(features.volatility, 0.0, 50.0))

        linear_combination = (
            self.w1 * spot_delta
            + self.w2 * book_imbalance
            - self.w3 * vol
        )
        return _sigmoid(linear_combination)

    def update_posterior(self, likelihood: float) -> float:
        """
        Apply Bayes' theorem to compute P(H|D).

        Uses the model's current self.prior as P(H).
        Does NOT mutate self.prior (use update_prior for online learning).

        Parameters
        ----------
        likelihood : float
            P(D|H) — typically the output of compute_likelihood.

        Returns
        -------
        float in (0, 1)
            Posterior probability P(H|D).
        """
        likelihood = float(np.clip(likelihood, 1e-12, 1.0 - 1e-12))
        prior = float(np.clip(self.prior, 1e-12, 1.0 - 1e-12))

        p_d_given_h = likelihood
        p_d_given_not_h = 1.0 - likelihood   # complement assumption
        p_h = prior
        p_not_h = 1.0 - prior

        # Total probability (denominator)
        p_d = p_d_given_h * p_h + p_d_given_not_h * p_not_h

        if p_d < 1e-300:
            # Degenerate case: return prior unchanged
            return prior

        posterior = (p_d_given_h * p_h) / p_d
        # Final numerical guard
        return float(np.clip(posterior, 1e-12, 1.0 - 1e-12))

    def compute_signal(self, features: MarketFeatures) -> BayesianSignal:
        """
        Full inference pipeline: features -> BayesianSignal.

        Parameters
        ----------
        features : MarketFeatures

        Returns
        -------
        BayesianSignal
        """
        likelihood = self.compute_likelihood(features)
        posterior = self.update_posterior(likelihood)

        # Log-odds: log(p / (1-p))
        posterior_clipped = float(np.clip(posterior, 1e-12, 1.0 - 1e-12))
        log_odds = math.log(posterior_clipped / (1.0 - posterior_clipped))

        # Signal strength: posterior itself serves as a [0,1] directional signal.
        # 0.5 = neutral, >0.5 = bullish mispricing, <0.5 = bearish mispricing.
        signal_strength = posterior

        return BayesianSignal(
            posterior_prob=posterior,
            likelihood=likelihood,
            prior=self.prior,
            log_odds=log_odds,
            signal_strength=signal_strength,
        )

    # ------------------------------------------------------------------
    # Online learning
    # ------------------------------------------------------------------

    def update_prior(self, outcome: bool, learning_rate: float = 0.05) -> None:
        """
        Online Bayesian prior update based on realised outcome.

        Moves self.prior toward 1.0 if the market was indeed mispriced
        (outcome=True) and toward 0.0 otherwise.

        Parameters
        ----------
        outcome      : bool  — True if mispricing was confirmed ex-post.
        learning_rate: float — Step size in [0, 1].  Default 0.05.
        """
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be in [0, 1], got {learning_rate!r}"
            )

        target = 1.0 if outcome else 0.0
        new_prior = self.prior + learning_rate * (target - self.prior)
        # Keep prior strictly inside (0, 1) to avoid degenerate inference
        self.prior = float(np.clip(new_prior, 1e-6, 1.0 - 1e-6))

    def reset_prior(self) -> None:
        """Reset self.prior to the value supplied at construction (default 0.5)."""
        self.prior = self._initial_prior

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BayesianModel(prior={self.prior:.4f}, "
            f"w1={self.w1}, w2={self.w2}, w3={self.w3})"
        )

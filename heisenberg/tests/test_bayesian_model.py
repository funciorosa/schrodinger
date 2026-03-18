"""
HEISENBERG — Bayesian Model Test Suite
Project: Polymarket Arbitrage Bot
Agent: BAYESIAN_AGENT

Run with:  pytest heisenberg/tests/test_bayesian_model.py -v
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — allows running tests from the repo root without installing.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from heisenberg.bayesian_model import (
    BayesianModel,
    BayesianSignal,
    MarketFeatures,
    compute_ewma_vol,
    _sigmoid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model() -> BayesianModel:
    """Default BayesianModel with prior=0.5."""
    return BayesianModel(prior=0.5)


@pytest.fixture
def neutral_features() -> MarketFeatures:
    """Market features representing a perfectly neutral / no-signal state."""
    return MarketFeatures(
        spot_delta=0.0,
        volatility=0.0,
        book_imbalance=0.0,
        spread=0.01,
        mid_price=0.5,
    )


@pytest.fixture
def bullish_features() -> MarketFeatures:
    """Strong positive signal: large spot_delta, buy-side book imbalance, low vol."""
    return MarketFeatures(
        spot_delta=2.0,
        volatility=0.01,
        book_imbalance=0.8,
        spread=0.02,
        mid_price=0.45,
    )


@pytest.fixture
def bearish_features() -> MarketFeatures:
    """Strong negative signal: negative spot_delta, sell-side book, higher vol."""
    return MarketFeatures(
        spot_delta=-2.0,
        volatility=0.5,
        book_imbalance=-0.8,
        spread=0.03,
        mid_price=0.55,
    )


# ---------------------------------------------------------------------------
# 1. Likelihood is always in [0, 1]
# ---------------------------------------------------------------------------

class TestLikelihoodRange:
    """Likelihood P(D|H) must stay within [0, 1] for all valid and edge-case inputs."""

    def test_likelihood_neutral(self, model: BayesianModel, neutral_features: MarketFeatures) -> None:
        likelihood = model.compute_likelihood(neutral_features)
        assert 0.0 <= likelihood <= 1.0

    def test_likelihood_bullish(self, model: BayesianModel, bullish_features: MarketFeatures) -> None:
        likelihood = model.compute_likelihood(bullish_features)
        assert 0.0 <= likelihood <= 1.0

    def test_likelihood_bearish(self, model: BayesianModel, bearish_features: MarketFeatures) -> None:
        likelihood = model.compute_likelihood(bearish_features)
        assert 0.0 <= likelihood <= 1.0

    @pytest.mark.parametrize("spot_delta,vol,imb", [
        (1e6, 1e6, 1.0),
        (-1e6, 0.0, -1.0),
        (0.0, 1e-9, 0.0),
        (float("inf"), 0.0, 0.0),   # should be clipped, not crash
    ])
    def test_likelihood_extreme_inputs(
        self,
        model: BayesianModel,
        spot_delta: float,
        vol: float,
        imb: float,
    ) -> None:
        features = MarketFeatures(
            spot_delta=spot_delta,
            volatility=vol,
            book_imbalance=imb,
            spread=0.01,
            mid_price=0.5,
        )
        likelihood = model.compute_likelihood(features)
        assert 0.0 <= likelihood <= 1.0
        assert math.isfinite(likelihood)


# ---------------------------------------------------------------------------
# 2. Posterior update follows Bayes' theorem
# ---------------------------------------------------------------------------

class TestPosteriorUpdate:
    """Verify P(H|D) is computed correctly from Bayes' rule."""

    def test_posterior_range(self, model: BayesianModel) -> None:
        for likelihood in [0.1, 0.3, 0.5, 0.7, 0.9]:
            posterior = model.update_posterior(likelihood)
            assert 0.0 < posterior < 1.0

    def test_posterior_manual_calculation(self, model: BayesianModel) -> None:
        """Cross-check against manual Bayes formula."""
        likelihood = 0.8
        prior = 0.5

        # Manual calculation
        p_d_given_h = likelihood
        p_d_given_not_h = 1.0 - likelihood
        p_d = p_d_given_h * prior + p_d_given_not_h * (1.0 - prior)
        expected_posterior = (p_d_given_h * prior) / p_d

        model.prior = prior
        computed_posterior = model.update_posterior(likelihood)
        assert abs(computed_posterior - expected_posterior) < 1e-9

    def test_posterior_high_likelihood_raises_posterior(self, model: BayesianModel) -> None:
        """A likelihood > 0.5 with prior=0.5 should push posterior above 0.5."""
        posterior = model.update_posterior(0.9)
        assert posterior > 0.5

    def test_posterior_low_likelihood_lowers_posterior(self, model: BayesianModel) -> None:
        """A likelihood < 0.5 with prior=0.5 should pull posterior below 0.5."""
        posterior = model.update_posterior(0.1)
        assert posterior < 0.5

    def test_posterior_likelihood_half_returns_prior(self, model: BayesianModel) -> None:
        """
        With likelihood=0.5 and prior=0.5:
            P(D|H)=0.5, P(D|~H)=0.5, P(D)=0.5
            posterior = 0.5 * 0.5 / 0.5 = 0.5
        """
        posterior = model.update_posterior(0.5)
        assert abs(posterior - 0.5) < 1e-9

    def test_posterior_biased_prior(self) -> None:
        """With a strong prior and high likelihood the posterior is very high."""
        model = BayesianModel(prior=0.9)
        posterior = model.update_posterior(0.95)
        assert posterior > 0.9   # should be higher than the prior

    def test_posterior_is_finite(self, model: BayesianModel) -> None:
        for likelihood in [1e-10, 0.5, 1.0 - 1e-10]:
            posterior = model.update_posterior(likelihood)
            assert math.isfinite(posterior)


# ---------------------------------------------------------------------------
# 3. Signal strength correlates with likelihood
# ---------------------------------------------------------------------------

class TestSignalStrength:
    """signal_strength must be monotonically related to likelihood (both equal posterior)."""

    def test_signal_strength_equals_posterior(
        self, model: BayesianModel, neutral_features: MarketFeatures
    ) -> None:
        signal = model.compute_signal(neutral_features)
        assert abs(signal.signal_strength - signal.posterior_prob) < 1e-12

    def test_signal_strength_increases_with_likelihood(self, model: BayesianModel) -> None:
        """
        A bullish feature set should produce higher signal_strength than a
        bearish one, given a symmetric prior.
        """
        bullish = MarketFeatures(
            spot_delta=3.0, volatility=0.01, book_imbalance=0.9,
            spread=0.01, mid_price=0.4,
        )
        bearish = MarketFeatures(
            spot_delta=-3.0, volatility=0.01, book_imbalance=-0.9,
            spread=0.01, mid_price=0.6,
        )
        sig_bull = model.compute_signal(bullish)
        sig_bear = model.compute_signal(bearish)
        assert sig_bull.signal_strength > sig_bear.signal_strength

    def test_signal_returns_bayesian_signal_type(
        self, model: BayesianModel, neutral_features: MarketFeatures
    ) -> None:
        signal = model.compute_signal(neutral_features)
        assert isinstance(signal, BayesianSignal)

    def test_signal_log_odds_sign_matches_posterior(
        self, model: BayesianModel, bullish_features: MarketFeatures
    ) -> None:
        """Positive posterior > 0.5 should produce positive log_odds."""
        signal = model.compute_signal(bullish_features)
        if signal.posterior_prob > 0.5:
            assert signal.log_odds > 0.0
        elif signal.posterior_prob < 0.5:
            assert signal.log_odds < 0.0

    def test_signal_log_odds_is_finite(
        self, model: BayesianModel, neutral_features: MarketFeatures
    ) -> None:
        signal = model.compute_signal(neutral_features)
        assert math.isfinite(signal.log_odds)


# ---------------------------------------------------------------------------
# 4. EWMA volatility
# ---------------------------------------------------------------------------

class TestEWMAVol:
    """Validate the compute_ewma_vol helper."""

    def test_ewma_vol_positive_for_nontrivial_series(self) -> None:
        prices = [100.0, 101.0, 99.5, 102.0, 100.5, 103.0, 101.5, 99.0, 104.0, 100.0]
        vol = compute_ewma_vol(prices)
        assert vol > 0.0

    def test_ewma_vol_zero_for_flat_series(self) -> None:
        prices = [100.0] * 50
        vol = compute_ewma_vol(prices)
        assert vol == 0.0

    def test_ewma_vol_single_price_returns_zero(self) -> None:
        assert compute_ewma_vol([100.0]) == 0.0

    def test_ewma_vol_empty_returns_zero(self) -> None:
        assert compute_ewma_vol([]) == 0.0

    def test_ewma_vol_is_finite_and_nonnegative(self) -> None:
        prices = list(np.random.default_rng(42).uniform(50, 150, 100))
        vol = compute_ewma_vol(prices)
        assert math.isfinite(vol)
        assert vol >= 0.0

    def test_ewma_vol_higher_span_smoother(self) -> None:
        """A larger span should produce a less reactive (smoother) vol estimate.
        This is a weak test: we just check both are positive and finite."""
        prices = [100.0 + math.sin(i * 0.3) * 5 for i in range(60)]
        vol_short = compute_ewma_vol(prices, span=5)
        vol_long = compute_ewma_vol(prices, span=50)
        assert vol_short > 0.0
        assert vol_long > 0.0
        assert math.isfinite(vol_short)
        assert math.isfinite(vol_long)

    def test_ewma_vol_minimum_two_prices(self) -> None:
        vol = compute_ewma_vol([100.0, 101.0])
        assert vol > 0.0


# ---------------------------------------------------------------------------
# 5. Extreme inputs do not produce NaN or Inf
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Stress-test all public methods with extreme / boundary values."""

    @pytest.mark.parametrize("spot_delta,vol,imb,spread,mid", [
        (1e15, 1e15, 1.0,  0.0,  1.0),
        (-1e15, 1e15, -1.0, 0.0, 0.0),
        (0.0,  0.0,  0.0,  0.0,  0.5),
        (1e-300, 1e-300, 0.0, 1e-300, 0.5),
    ])
    def test_compute_signal_no_nan_inf(
        self, spot_delta: float, vol: float, imb: float, spread: float, mid: float
    ) -> None:
        model = BayesianModel()
        features = MarketFeatures(
            spot_delta=spot_delta,
            volatility=vol,
            book_imbalance=imb,
            spread=spread,
            mid_price=mid,
        )
        sig = model.compute_signal(features)
        assert math.isfinite(sig.posterior_prob)
        assert math.isfinite(sig.likelihood)
        assert math.isfinite(sig.log_odds)
        assert math.isfinite(sig.signal_strength)

    def test_posterior_no_nan_near_boundary_likelihoods(self) -> None:
        model = BayesianModel()
        for likelihood in [1e-15, 1e-10, 0.5, 1.0 - 1e-10, 1.0 - 1e-15]:
            posterior = model.update_posterior(likelihood)
            assert math.isfinite(posterior)
            assert 0.0 < posterior < 1.0

    def test_model_survives_repeated_updates(self) -> None:
        model = BayesianModel(prior=0.5)
        features = MarketFeatures(1.0, 0.1, 0.3, 0.02, 0.5)
        for _ in range(1_000):
            sig = model.compute_signal(features)
            assert math.isfinite(sig.posterior_prob)


# ---------------------------------------------------------------------------
# 6. Online learning (update_prior)
# ---------------------------------------------------------------------------

class TestOnlineLearning:
    """update_prior must move self.prior in the correct direction."""

    def test_prior_increases_on_positive_outcome(self, model: BayesianModel) -> None:
        initial_prior = model.prior
        model.update_prior(outcome=True)
        assert model.prior > initial_prior

    def test_prior_decreases_on_negative_outcome(self, model: BayesianModel) -> None:
        initial_prior = model.prior
        model.update_prior(outcome=False)
        assert model.prior < initial_prior

    def test_prior_converges_toward_one_after_many_true_outcomes(self) -> None:
        model = BayesianModel(prior=0.5)
        for _ in range(200):
            model.update_prior(outcome=True, learning_rate=0.1)
        assert model.prior > 0.95

    def test_prior_converges_toward_zero_after_many_false_outcomes(self) -> None:
        model = BayesianModel(prior=0.5)
        for _ in range(200):
            model.update_prior(outcome=False, learning_rate=0.1)
        assert model.prior < 0.05

    def test_prior_stays_in_open_unit_interval(self) -> None:
        model = BayesianModel(prior=0.5)
        for _ in range(500):
            model.update_prior(outcome=True, learning_rate=0.9)
        assert 0.0 < model.prior < 1.0

    def test_prior_zero_learning_rate_no_change(self, model: BayesianModel) -> None:
        initial_prior = model.prior
        model.update_prior(outcome=True, learning_rate=0.0)
        assert model.prior == initial_prior

    def test_invalid_learning_rate_raises(self, model: BayesianModel) -> None:
        with pytest.raises(ValueError):
            model.update_prior(outcome=True, learning_rate=1.5)

    def test_invalid_prior_raises(self) -> None:
        with pytest.raises(ValueError):
            BayesianModel(prior=0.0)
        with pytest.raises(ValueError):
            BayesianModel(prior=1.0)
        with pytest.raises(ValueError):
            BayesianModel(prior=-0.1)

    def test_reset_prior_restores_initial(self) -> None:
        model = BayesianModel(prior=0.3)
        model.update_prior(outcome=True, learning_rate=0.5)
        assert model.prior != 0.3
        model.reset_prior()
        assert abs(model.prior - 0.3) < 1e-12


# ---------------------------------------------------------------------------
# 7. Zero book imbalance gives symmetric result
# ---------------------------------------------------------------------------

class TestSymmetry:
    """With book_imbalance=0 and symmetric spot_delta, the model should behave symmetrically."""

    def test_zero_imbalance_neutral_spot_delta(self, model: BayesianModel) -> None:
        """spot_delta=0, imbalance=0, vol=0 => likelihood = sigmoid(0) = 0.5."""
        features = MarketFeatures(
            spot_delta=0.0, volatility=0.0, book_imbalance=0.0,
            spread=0.01, mid_price=0.5,
        )
        likelihood = model.compute_likelihood(features)
        assert abs(likelihood - 0.5) < 1e-9

    def test_zero_imbalance_symmetric_spot_delta(self, model: BayesianModel) -> None:
        """
        With imbalance=0, vol=0:
          likelihood(+d) = sigmoid(w1 * d)
          likelihood(-d) = sigmoid(-w1 * d)
          They should sum to 1 (sigmoid symmetry).
        """
        d = 1.5
        f_pos = MarketFeatures(spot_delta=d,  volatility=0.0, book_imbalance=0.0, spread=0.01, mid_price=0.5)
        f_neg = MarketFeatures(spot_delta=-d, volatility=0.0, book_imbalance=0.0, spread=0.01, mid_price=0.5)

        l_pos = model.compute_likelihood(f_pos)
        l_neg = model.compute_likelihood(f_neg)

        assert abs(l_pos + l_neg - 1.0) < 1e-9

    def test_symmetry_posterior_sums_to_one_approximately(self, model: BayesianModel) -> None:
        """
        Given symmetric likelihoods and prior=0.5, posteriors should also be symmetric:
          posterior(l) + posterior(1-l) ≈ 1.
        """
        likelihood = 0.75
        p1 = model.update_posterior(likelihood)
        p2 = model.update_posterior(1.0 - likelihood)
        assert abs(p1 + p2 - 1.0) < 1e-9

    def test_zero_imbalance_pure_vol_reduces_likelihood(self, model: BayesianModel) -> None:
        """Higher volatility should reduce the likelihood (negative weight -w3)."""
        low_vol = MarketFeatures(spot_delta=0.0, volatility=0.1, book_imbalance=0.0, spread=0.01, mid_price=0.5)
        high_vol = MarketFeatures(spot_delta=0.0, volatility=5.0, book_imbalance=0.0, spread=0.01, mid_price=0.5)

        l_low = model.compute_likelihood(low_vol)
        l_high = model.compute_likelihood(high_vol)

        assert l_low > l_high


# ---------------------------------------------------------------------------
# 8. Integration: full pipeline consistency
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end tests confirming all components work together coherently."""

    def test_compute_signal_prior_field_matches_model_prior(self, model: BayesianModel) -> None:
        features = MarketFeatures(0.5, 0.1, 0.2, 0.02, 0.5)
        signal = model.compute_signal(features)
        assert signal.prior == model.prior

    def test_reset_prior_after_learning_restores_inference(self) -> None:
        model = BayesianModel(prior=0.5)
        features = MarketFeatures(1.0, 0.1, 0.3, 0.02, 0.5)

        signal_before = model.compute_signal(features)

        # Simulate learning
        for _ in range(50):
            model.update_prior(outcome=True, learning_rate=0.1)

        signal_biased = model.compute_signal(features)
        assert signal_biased.posterior_prob != signal_before.posterior_prob

        model.reset_prior()
        signal_after_reset = model.compute_signal(features)
        assert abs(signal_after_reset.posterior_prob - signal_before.posterior_prob) < 1e-12

    def test_high_spot_delta_and_buy_imbalance_gives_strong_signal(self) -> None:
        model = BayesianModel(prior=0.5)
        features = MarketFeatures(
            spot_delta=4.0, volatility=0.01, book_imbalance=1.0,
            spread=0.01, mid_price=0.4,
        )
        signal = model.compute_signal(features)
        # Expect a strong bullish signal
        assert signal.signal_strength > 0.85
        assert signal.log_odds > 0.0

    def test_model_repr_contains_prior(self) -> None:
        model = BayesianModel(prior=0.3)
        assert "0.3" in repr(model)

    def test_ewma_vol_integrated_with_model(self) -> None:
        """Use compute_ewma_vol output as volatility input to the model."""
        prices = [100.0, 102.0, 99.0, 101.5, 98.5, 103.0, 100.5]
        vol = compute_ewma_vol(prices)
        assert vol >= 0.0

        model = BayesianModel()
        features = MarketFeatures(
            spot_delta=0.5,
            volatility=vol,
            book_imbalance=0.3,
            spread=0.02,
            mid_price=0.5,
        )
        signal = model.compute_signal(features)
        assert math.isfinite(signal.posterior_prob)
        assert 0.0 < signal.posterior_prob < 1.0

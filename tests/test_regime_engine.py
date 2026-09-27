"""
Synthetic, network-free regression tests for the regime-engine fixes:

B3/B4/B5 -- covariance regularization (diag/shrunk vs full), causal
           standardization of HMM inputs, and the window-size guardrail.
B1/B6    -- predictive (not in-sample) conditional moments, and the
           predicted-next-day belief driving confidence/forecasting.
C1/C3    -- holding-period-scaled objective, smooth QP with no silent
           equal-weight fallback on infeasible input.

Uses synthetic factor-return data throughout (no network, no dependency
on an already-calibrated risk model) so these can run fast and
deterministically, independent of tests/test_regime_api.py's live E2E path.
"""
import unittest

import numpy as np
import pandas as pd

from oport.functions.regime.regime_engine import (
    RegimeModelConfig,
    RegimeCalibrator,
    PortfolioConstructor,
    n_hmm_params,
    _causal_standardize,
)


def _make_synthetic_factor_returns(n_days=400, seed=0):
    """Two well-separated regimes on a 2-factor synthetic series, switching
    every ~60 days -- enough separation for a diag-covariance 2-state HMM
    to recover cleanly without needing real market data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    regime_len = 60
    rows = []
    state = 0
    for i in range(n_days):
        if i % regime_len == 0:
            state = 1 - state
        if state == 0:
            mu, sd = np.array([0.0010, 0.0005]), np.array([0.010, 0.008])
        else:
            mu, sd = np.array([-0.0020, -0.0010]), np.array([0.030, 0.025])
        rows.append(rng.normal(mu, sd))
    return pd.DataFrame(rows, index=dates, columns=["f1", "f2"])


def _synthetic_cfg(**overrides):
    params = dict(
        min_regimes=2, max_regimes=2, cov_type="diag",
        estimation_window=150, order_selection_holdout=21,
        refit_frequency=30, min_params_multiplier=5.0,
    )
    params.update(overrides)
    return RegimeModelConfig(**params)


class TestGaussianHMMFixes(unittest.TestCase):

    def test_n_hmm_params_diag_vs_full(self):
        self.assertEqual(n_hmm_params(d=3, k=2, cov_type="diag"), 2 * 3 + 2 * 3 + 2 * 1 + 1)
        self.assertEqual(n_hmm_params(d=3, k=2, cov_type="full"), 2 * 3 + 2 * (3 * 4 // 2) + 2 * 1 + 1)
        self.assertGreater(
            n_hmm_params(d=9, k=4, cov_type="full"),
            n_hmm_params(d=9, k=4, cov_type="diag"),
        )

    def test_causal_standardize_uses_only_trailing_history(self):
        fr = _make_synthetic_factor_returns(n_days=200)
        z = _causal_standardize(fr, halflife=63, min_periods=2)

        fr_perturbed = fr.copy()
        fr_perturbed.iloc[150] *= 5.0
        z_perturbed = _causal_standardize(fr_perturbed, halflife=63, min_periods=2)

        # A shock at row 150 must not move any EARLIER row's z-score.
        pd.testing.assert_frame_equal(z.iloc[:100], z_perturbed.iloc[:100])
        # ... but should move later ones (positive control).
        self.assertFalse(z.iloc[150:].equals(z_perturbed.iloc[150:]))

    def test_undersized_window_is_rejected(self):
        fr = _make_synthetic_factor_returns(n_days=100)
        cfg = _synthetic_cfg(
            estimation_window=30, order_selection_holdout=5, refit_frequency=10,
            min_params_multiplier=10.0,  # default guardrail, not the relaxed test value
        )
        cal = RegimeCalibrator(cfg)
        cal.load_factor_returns(fr)
        with self.assertRaises(ValueError):
            cal.run()

    def test_adequately_sized_window_is_accepted(self):
        fr = _make_synthetic_factor_returns(n_days=400)
        cal = RegimeCalibrator(_synthetic_cfg())
        cal.load_factor_returns(fr)
        cal.run()
        self.assertIsNotNone(cal.current_regime_)


class TestInitializeRegimeModelDefaultWindow(unittest.TestCase):
    """functions_regime.initialize_regime_model's estimation_window default
    must actually satisfy its OWN min_params_multiplier guard for the real
    factor count -- a fixed estimation_window=252 default silently
    violated that guard for any realistic (~9-dimensional) continuous
    factor set regardless of how much history was loaded, which is
    exactly the mismatch that pushed a prior run to relax
    min_params_multiplier to 8.5 instead of fixing the real problem."""

    def test_default_estimation_window_satisfies_its_own_guard(self):
        from oport.functions.functions_regime import initialize_regime_model

        fr = _make_synthetic_factor_returns(n_days=450, seed=2)
        cal, cfg = initialize_regime_model(factor_returns=fr)  # no estimation_window override

        worst_case_params = n_hmm_params(fr.shape[1], cfg.max_regimes, cfg.cov_type)
        usable_days = cfg.estimation_window - cfg.order_selection_holdout
        self.assertGreaterEqual(
            usable_days, cfg.min_params_multiplier * worst_case_params,
            "Derived default estimation_window doesn't satisfy its own guard.",
        )
        self.assertIsNotNone(cal.current_regime_)


class TestLowMassFallbackStaysInRawUnits(unittest.TestCase):
    """Regression test for the template-unit bug: when a persistent
    regime's belief-weighted mass drops below the w_sum>=5 threshold in a
    given calibration, the fallback must reuse the persistent RAW-scale
    estimate from an earlier calibration -- NOT the Wasserstein template
    book's mean/cov, which live in Z-SCORED (standardized) units. Mixing
    the two previously let an O(1) z-scored value silently stand in for an
    O(1e-4..1e-3) raw daily factor mean."""

    def test_fallback_uses_raw_estimate_not_zscored_template(self):
        cfg = _synthetic_cfg()
        cal = RegimeCalibrator(cfg)

        dates = pd.bdate_range("2020-01-01", periods=30)
        fr = pd.DataFrame(
            {"f1": np.full(30, 0.001), "f2": np.full(30, 0.0005)}, index=dates
        )
        # Regime 0 has ample belief mass; regime 1 has only ~0.6 days of
        # mass (w_sum < 5) -- exactly the "~1 day of mass" scenario from
        # the bug report.
        prob_df = pd.DataFrame(
            {0: np.full(30, 0.98), 1: np.full(30, 0.02)}, index=dates
        )

        # Seed a persistent raw-scale estimate for regime 1, as an earlier
        # (real) calibration with adequate mass would have.
        raw_mean_1 = pd.Series([0.0020, -0.0010], index=["f1", "f2"])
        cal._raw_regime_mean[1] = raw_mean_1
        cal._raw_regime_cov[1] = pd.DataFrame(
            np.eye(2) * 1e-4, index=["f1", "f2"], columns=["f1", "f2"]
        )
        # And a z-scored Wasserstein template for regime 1 -- O(1) scale,
        # the value the old code incorrectly fell back to.
        cal._template_book.means[1] = np.array([0.5, -0.3])
        cal._template_book.covs[1] = np.eye(2)

        cal._fit_conditional_moments(fr, prob_df, [0, 1])

        pd.testing.assert_series_equal(cal.regime_mean_[1], raw_mean_1)
        self.assertLess(
            float(cal.regime_mean_[1].abs().max()), 0.01,
            "regime_mean_[1] should be a raw daily factor mean, not the "
            "O(1) z-scored template value.",
        )


class TestRegimeConditionalMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fr = _make_synthetic_factor_returns(n_days=400)
        cls.cal = RegimeCalibrator(_synthetic_cfg())
        cls.cal.load_factor_returns(cls.fr)
        cls.cal.run()

    def test_regime_mean_is_not_the_in_sample_label_average(self):
        """The in-sample average of a regime's own labeled days is exactly
        what B1 flagged as biased -- confirm regime_mean_ is NOT silently
        still that formula."""
        checked_any = False
        for rid, mean in self.cal.regime_mean_.items():
            days = self.cal.regime_label_[self.cal.regime_label_ == rid].index
            if len(days) < 5:
                continue
            in_sample_avg = self.fr.loc[self.fr.index.isin(days)].mean()
            self.assertFalse(mean.equals(in_sample_avg))
            checked_any = True
        self.assertTrue(checked_any, "No regime had enough labeled days to check.")

    def test_predicted_next_proba_is_a_valid_distribution(self):
        pi_next = self.cal.predicted_next_proba()
        self.assertAlmostEqual(float(pi_next.sum()), 1.0, places=6)
        self.assertTrue((pi_next >= -1e-9).all())

    def test_conditional_moments_none_blends_by_forecast(self):
        mean_blend, _ = self.cal.conditional_moments(None)
        mean_hard, _ = self.cal.conditional_moments(self.cal.current_regime_)
        pi_next = self.cal.predicted_next_proba()
        if pi_next.max() < 0.999:
            self.assertFalse(mean_blend.equals(mean_hard))


class TestPortfolioConstructorFixes(unittest.TestCase):

    def _toy_inputs(self, n=4):
        tickers = [f"S{i}" for i in range(n)]
        factors = ["f1", "f2"]
        exposures = pd.DataFrame(
            np.random.default_rng(1).normal(size=(n, 2)),
            index=tickers, columns=factors,
        )
        factor_cov = pd.DataFrame(
            [[0.02, 0.005], [0.005, 0.015]], index=factors, columns=factors,
        )
        idio_var = pd.Series(0.01, index=tickers)
        regime_factor_mean = pd.Series({"f1": 0.001, "f2": 0.0005})
        return exposures, factor_cov, idio_var, regime_factor_mean, tickers

    def test_weights_sum_to_one_and_respect_bounds(self):
        exposures, factor_cov, idio_var, mu_f, tickers = self._toy_inputs()
        pc = PortfolioConstructor(RegimeModelConfig(max_weight=0.5))
        out = pc.construct(
            exposures=exposures, factor_cov=factor_cov, idio_var=idio_var,
            regime_factor_mean=mu_f, regime_id=0, regime_confidence=0.9,
            expected_hold_days=10.0, universe=tickers,
        )
        self.assertAlmostEqual(out["weight"].sum(), 1.0, places=4)
        self.assertTrue((out["weight"] <= 0.5 + 1e-6).all())
        self.assertTrue((out["weight"] >= -1e-9).all())

    def test_holding_period_changes_optimal_tilt(self):
        """C1: a longer expected holding period should let expected return
        dominate the one-off transaction cost more, producing at least as
        much turnover away from the equal-weight start as a 1-day horizon."""
        exposures, factor_cov, idio_var, mu_f, tickers = self._toy_inputs()
        cfg = RegimeModelConfig(max_weight=1.0, transaction_cost_bps=50.0)
        pc = PortfolioConstructor(cfg)
        w_prev = pd.Series(1.0 / len(tickers), index=tickers)

        out_short = pc.construct(
            exposures=exposures, factor_cov=factor_cov, idio_var=idio_var,
            regime_factor_mean=mu_f, regime_id=0, regime_confidence=0.9,
            expected_hold_days=1.0, universe=tickers, current_weights=w_prev,
        )
        out_long = pc.construct(
            exposures=exposures, factor_cov=factor_cov, idio_var=idio_var,
            regime_factor_mean=mu_f, regime_id=0, regime_confidence=0.9,
            expected_hold_days=60.0, universe=tickers, current_weights=w_prev,
        )
        turnover_short = (out_short["weight"] - 1.0 / len(tickers)).abs().sum()
        turnover_long = (out_long["weight"] - 1.0 / len(tickers)).abs().sum()
        self.assertGreaterEqual(turnover_long + 1e-9, turnover_short)

    def test_infeasible_constraints_raise_instead_of_silently_equal_weighting(self):
        """C3: max_weight too small to sum to 1 must raise, not fall back
        to equal weights the way the old `w_opt = result.x if result.success
        else w0` did."""
        exposures, factor_cov, idio_var, mu_f, tickers = self._toy_inputs(n=4)
        pc = PortfolioConstructor(RegimeModelConfig(max_weight=0.1))  # 4*0.1=0.4 < 1
        with self.assertRaises(RuntimeError):
            pc.construct(
                exposures=exposures, factor_cov=factor_cov, idio_var=idio_var,
                regime_factor_mean=mu_f, regime_id=0, regime_confidence=0.9,
                expected_hold_days=10.0, universe=tickers,
            )


if __name__ == "__main__":
    unittest.main()

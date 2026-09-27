"""
regime_engine.py
=================
Wasserstein Hidden Markov Model regime inference + regime-conditioned,
transaction-cost-aware portfolio construction.

Mirrors the shape of `risk.risk_engine` (ModelConfig / DailyCalibrator) so
it can be dropped into the same suite:

    RegimeModelConfig   <->  ModelConfig
    RegimeCalibrator    <->  DailyCalibrator
    PortfolioConstructor is new: it consumes a RegimeCalibrator (regime
    probabilities / conditional factor moments) together with an already
    calibrated Barra-style risk model (factor exposures + factor
    covariance from risk_engine) to produce stock-level target weights
    and a suggested holding period per name.

Design (per paper "Explainable Regime-Aware Investing", Boukardagha 2026,
arXiv:2603.04441):
  1. Strictly causal rolling Gaussian HMM estimation (walk-forward, no
     look-ahead: every regime label / probability assigned to day t only
     uses data up to and including t).
  2. Predictive model-order selection: the number of regimes K is chosen
     at each refit by one-step-ahead out-of-sample log-likelihood on a
     holdout window, not in-sample fit.
  3. Wasserstein identity tracking: because re-fitting an HMM from
     scratch produces an arbitrary permutation of state labels, each new
     window's Gaussian components are matched to a set of persistent
     "regime templates" via the 2-Wasserstein distance (solved with the
     Hungarian algorithm), and templates are updated with an EMA. This
     keeps "Regime 1" meaning the same thing across refits.

Regimes are inferred at the cross-asset / factor level (this
implementation is fed the risk model's own factor-return time series,
not raw stock returns), and then mapped down to individual names via
each stock's factor exposures (betas) from the Barra-style risk model,
so a single market-wide regime call is what drives stock-level expected
returns -- not a separate per-name regime.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import logsumexp
from scipy.cluster.vq import kmeans2

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class RegimeModelConfig:
    """Mirrors risk_engine.ModelConfig's role: one place for every knob."""

    # --- regime order selection ---
    min_regimes: int = 2
    max_regimes: int = 4
    order_selection_holdout: int = 21          # trading days held out for
                                                 # one-step-ahead predictive log-lik

    # --- rolling / walk-forward estimation ---
    estimation_window: int = 252                # lookback window per fit (trading days)
    refit_frequency: int = 21                    # trading days between HMM refits

    # --- HMM input standardization ---
    # Factor volatilities can differ by 2x or more, and both the Gaussian
    # likelihood and the Wasserstein template matching are scale-sensitive.
    # Each factor is z-scored by ITS OWN trailing (causal, no full-sample)
    # volatility before being fed to the HMM -- see _causal_standardize.
    standardize_halflife: int = 63
    standardize_min_periods: int = 2

    # --- EM ---
    em_max_iter: int = 150
    em_tol: float = 1e-4
    # Relative diagonal loading for numerical stability: the actual epsilon
    # added to each state's covariance is cov_reg_relative * mean(diag(cov(X)))
    # for the window being fit, not a fixed absolute number -- factor-return
    # scales vary across windows and factors, so a fixed 1e-6 is too small
    # to regularize a near-singular fit on some windows and pointless on others.
    cov_reg_relative: float = 1e-2
    # "diag": zero out cross-factor covariance terms (default -- a k-state,
    #   d-dim full covariance has k*d*(d+1)/2 parameters, which a rolling
    #   window of a few hundred days cannot support without near-singular,
    #   overconfident states).
    # "shrunk": linear shrinkage of the full covariance toward its diagonal
    #   (intensity = shrinkage_intensity).
    # "full": unregularized sample covariance -- only safe with ample data.
    cov_type: str = "diag"
    shrinkage_intensity: float = 0.3             # used only when cov_type == "shrunk"
    n_em_restarts: int = 3
    random_state: int = 7
    # Guardrail: reject a fit whose (estimation_window - holdout) training
    # days can't support >= min_params_multiplier x the HMM's free
    # parameter count at max_regimes. Set to 0 to disable.
    min_params_multiplier: float = 10.0

    # --- identity tracking ---
    wasserstein_new_regime_threshold: Optional[float] = None
    # if set, a matched component whose Wasserstein distance to its
    # closest template exceeds this is treated as a *new* persistent
    # regime rather than forced onto an existing template
    template_ema_halflife: int = 63              # days, EMA smoothing of template mean/cov

    # --- portfolio construction ---
    risk_aversion: float = 5.0
    transaction_cost_bps: float = 10.0
    max_weight: float = 0.10
    long_only: bool = True
    target_n: Optional[int] = None               # cap on number of names held
    # Shrinkage of the regime-conditional tilt toward the factor's own
    # unconditional mean: mu_f = mu_uncond + alpha * (mu_regime - mu_uncond).
    # alpha=1 trusts the regime estimate fully; alpha=0 ignores it. Sector
    # dummies (which the regime model does not speak to at all) get a flat
    # 0 prior instead of their unconditional historical mean -- see C2.
    regime_tilt_alpha: float = 0.5


# --------------------------------------------------------------------------- #
# Small numerical helpers
# --------------------------------------------------------------------------- #

def _reg_cov(cov: np.ndarray, eps: float) -> np.ndarray:
    d = cov.shape[0]
    return cov + eps * np.eye(d)


def _relative_cov_reg(X: np.ndarray, cfg: "RegimeModelConfig") -> float:
    """Diagonal loading epsilon, scaled to this window's own factor
    variances rather than a fixed absolute constant (see cov_reg_relative)."""
    mean_var = float(np.mean(np.diag(np.cov(X.T))))
    return max(cfg.cov_reg_relative * mean_var, 1e-12)


def _apply_cov_type(cov: np.ndarray, cfg: "RegimeModelConfig") -> np.ndarray:
    """Regularize a raw (co)variance estimate per cfg.cov_type -- see the
    field docstring on RegimeModelConfig for the rationale."""
    if cfg.cov_type == "diag":
        return np.diag(np.diag(cov))
    if cfg.cov_type == "shrunk":
        target = np.diag(np.diag(cov))
        a = min(max(cfg.shrinkage_intensity, 0.0), 1.0)
        return (1.0 - a) * cov + a * target
    return cov


def _causal_standardize(fr: pd.DataFrame, halflife: int, min_periods: int) -> pd.DataFrame:
    """
    Z-score each factor by its own trailing (EWM, causal) volatility --
    row t only ever uses data up to and including t, never the full-sample
    std, which would leak the entire history's scale into every day's HMM
    input. Degenerate rows (insufficient history, zero trailing vol) are
    zeroed rather than left NaN/inf.
    """
    trailing_vol = fr.ewm(halflife=halflife, min_periods=min_periods).std()
    trailing_vol = trailing_vol.replace(0.0, np.nan)
    return (fr / trailing_vol).fillna(0.0)


def n_hmm_params(d: int, k: int, cov_type: str = "diag") -> int:
    """
    Number of free parameters in a k-state, d-dimensional Gaussian HMM,
    given the covariance parameterization. Used to size-check a rolling
    window before fitting -- see RegimeCalibrator.run / min_params_multiplier.
    """
    mean_params  = k * d
    cov_params   = k * d if cov_type == "diag" else k * d * (d + 1) // 2
    trans_params = k * (k - 1)
    start_params = k - 1
    return mean_params + cov_params + trans_params + start_params


def _mvn_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of a multivariate normal, vectorized over rows of X."""
    d = mean.shape[0]
    diff = X - mean
    sign, logdet = np.linalg.slogdet(cov)
    prec = np.linalg.inv(cov)
    quad = np.einsum("ij,jk,ik->i", diff, prec, diff)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def wasserstein2_gaussian(mean1: np.ndarray, cov1: np.ndarray,
                           mean2: np.ndarray, cov2: np.ndarray) -> float:
    """
    Closed-form squared 2-Wasserstein distance between two Gaussians:
        W2^2 = ||m1 - m2||^2 + tr(C1 + C2 - 2*(C2^{1/2} C1 C2^{1/2})^{1/2})
    Returns the (non-squared) distance.
    """
    mean_term = float(np.sum((mean1 - mean2) ** 2))
    c2_sqrt = sqrtm(cov2)
    c2_sqrt = np.real_if_close(c2_sqrt)
    inner = c2_sqrt @ cov1 @ c2_sqrt
    inner_sqrt = sqrtm(inner)
    inner_sqrt = np.real_if_close(inner_sqrt)
    cov_term = float(np.trace(cov1 + cov2 - 2 * inner_sqrt))
    cov_term = max(cov_term, 0.0)
    return float(np.sqrt(max(mean_term + cov_term, 0.0)))


# --------------------------------------------------------------------------- #
# Gaussian HMM (Baum-Welch EM in log-space)
# --------------------------------------------------------------------------- #

class _GaussianHMM:
    """Minimal, dependency-light Gaussian HMM with EM fitting and a
    filtered (causal, forward-only) inference pass. Not exposed outside
    this module -- RegimeCalibrator is the public surface."""

    def __init__(self, n_states: int, cfg: RegimeModelConfig):
        self.k = n_states
        self.cfg = cfg
        self.means_: Optional[np.ndarray] = None
        self.covs_: Optional[np.ndarray] = None
        self.trans_: Optional[np.ndarray] = None
        self.start_: Optional[np.ndarray] = None
        self.reg_eps_: float = 1e-12

    def _init_params(self, X: np.ndarray, seed: int, reg_eps: float):
        d = X.shape[1]
        centroid, label = kmeans2(X, self.k, seed=seed, minit="++")
        means = centroid.copy()
        covs = np.zeros((self.k, d, d))
        for j in range(self.k):
            pts = X[label == j]
            raw_cov = np.cov(X.T) if len(pts) < d + 2 else np.cov(pts.T)
            covs[j] = _reg_cov(_apply_cov_type(raw_cov, self.cfg), reg_eps)
        trans = np.full((self.k, self.k), 0.05 / max(self.k - 1, 1))
        np.fill_diagonal(trans, 0.95)
        start = np.full(self.k, 1.0 / self.k)
        return means, covs, trans, start

    def _log_emissions(self, X: np.ndarray, means, covs, reg_eps: float) -> np.ndarray:
        T = X.shape[0]
        logB = np.zeros((T, self.k))
        for j in range(self.k):
            logB[:, j] = _mvn_logpdf(X, means[j], _reg_cov(covs[j], reg_eps))
        return logB

    def _forward_backward(self, logB, log_trans, log_start):
        T, K = logB.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_start + logB[0]
        for t in range(1, T):
            log_alpha[t] = logB[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_trans, axis=0
            )
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_trans + logB[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
            )
        loglik = logsumexp(log_alpha[-1])
        log_gamma = log_alpha + log_beta - loglik
        gamma = np.exp(log_gamma)

        log_xi = np.full((T - 1, K, K), -np.inf)
        for t in range(T - 1):
            log_xi[t] = (
                log_alpha[t][:, None]
                + log_trans
                + logB[t + 1][None, :]
                + log_beta[t + 1][None, :]
                - loglik
            )
        xi = np.exp(log_xi)
        return gamma, xi, loglik

    def fit(self, X: np.ndarray) -> float:
        reg_eps = _relative_cov_reg(X, self.cfg)
        best_ll = -np.inf
        best_params = None
        for r in range(self.cfg.n_em_restarts):
            means, covs, trans, start = self._init_params(
                X, seed=self.cfg.random_state + r, reg_eps=reg_eps
            )
            prev_ll = -np.inf
            for it in range(self.cfg.em_max_iter):
                logB = self._log_emissions(X, means, covs, reg_eps)
                gamma, xi, loglik = self._forward_backward(
                    logB, np.log(trans + 1e-300), np.log(start + 1e-300)
                )
                if np.isnan(loglik) or np.isinf(loglik):
                    break
                if abs(loglik - prev_ll) < self.cfg.em_tol:
                    prev_ll = loglik
                    break
                prev_ll = loglik

                # M-step
                start = gamma[0] / gamma[0].sum()
                trans_num = xi.sum(axis=0)
                trans = trans_num / trans_num.sum(axis=1, keepdims=True)
                Nk = gamma.sum(axis=0)
                for j in range(self.k):
                    w = gamma[:, j]
                    means[j] = (w[:, None] * X).sum(axis=0) / max(Nk[j], 1e-8)
                    diff = X - means[j]
                    raw_cov = (
                        (w[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
                        / max(Nk[j], 1e-8)
                    )
                    covs[j] = _reg_cov(_apply_cov_type(raw_cov, self.cfg), reg_eps)
            if prev_ll > best_ll:
                best_ll = prev_ll
                best_params = (means.copy(), covs.copy(), trans.copy(), start.copy())

        self.means_, self.covs_, self.trans_, self.start_ = best_params
        self.reg_eps_ = reg_eps
        return best_ll

    def filtered_proba(self, X: np.ndarray) -> np.ndarray:
        """Causal (forward-only) filtered state probabilities: gamma_t
        depends only on X[0..t]. Used so no future information leaks
        into a given day's regime label."""
        logB = self._log_emissions(X, self.means_, self.covs_, self.reg_eps_)
        log_trans = np.log(self.trans_ + 1e-300)
        T, K = logB.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.start_ + 1e-300) + logB[0]
        for t in range(1, T):
            log_alpha[t] = logB[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_trans, axis=0
            )
        log_alpha -= logsumexp(log_alpha, axis=1, keepdims=True)
        return np.exp(log_alpha)

    def one_step_ahead_loglik(self, X_train: np.ndarray, X_holdout: np.ndarray) -> float:
        """Predictive log-likelihood of X_holdout, one step at a time,
        filtering forward and predicting the *next* observation's
        marginal density under the current belief before seeing it."""
        filt = self.filtered_proba(X_train)
        belief = filt[-1]
        total = 0.0
        X_ext = np.vstack([X_train[-1:], X_holdout])
        for t in range(1, len(X_ext)):
            pred_belief = belief @ self.trans_
            logB_t = np.array([
                _mvn_logpdf(X_ext[t:t + 1], self.means_[j],
                            _reg_cov(self.covs_[j], self.reg_eps_))[0]
                for j in range(self.k)
            ])
            mix_loglik = logsumexp(np.log(pred_belief + 1e-300) + logB_t)
            total += mix_loglik
            post = np.exp(np.log(pred_belief + 1e-300) + logB_t - mix_loglik)
            belief = post / post.sum()
        return total / max(len(X_holdout), 1)


def select_model_order(X: np.ndarray, cfg: RegimeModelConfig) -> Tuple[int, _GaussianHMM, float]:
    """Predictive model-order selection: fit each K on the training part
    of the window, score one-step-ahead predictive log-lik on the held
    out tail, pick the K with the best (highest) score."""
    holdout = cfg.order_selection_holdout
    if len(X) <= holdout + 30:
        holdout = max(5, len(X) // 5)
    X_train, X_hold = X[:-holdout], X[-holdout:]

    best_k, best_model, best_score = None, None, -np.inf
    for k in range(cfg.min_regimes, cfg.max_regimes + 1):
        model = _GaussianHMM(k, cfg)
        model.fit(X_train)
        score = model.one_step_ahead_loglik(X_train, X_hold)
        if score > best_score:
            best_k, best_model, best_score = k, model, score

    # refit the winning K on the FULL window (still causal: window end == t)
    final_model = _GaussianHMM(best_k, cfg)
    final_model.fit(X)
    return best_k, final_model, best_score


# --------------------------------------------------------------------------- #
# Wasserstein identity tracking
# --------------------------------------------------------------------------- #

class _RegimeTemplateBook:
    """Keeps persistent regime identities alive across refits."""

    def __init__(self, cfg: RegimeModelConfig):
        self.cfg = cfg
        self.next_id = 0
        self.means: Dict[int, np.ndarray] = {}
        self.covs: Dict[int, np.ndarray] = {}
        self.ema_alpha = 1 - 0.5 ** (1.0 / max(cfg.template_ema_halflife, 1))

    def match(self, new_means: np.ndarray, new_covs: np.ndarray) -> Dict[int, int]:
        """Returns {local_state_index -> persistent_regime_id}."""
        k_new = new_means.shape[0]
        if not self.means:
            mapping = {}
            for j in range(k_new):
                rid = self.next_id
                self.next_id += 1
                self.means[rid] = new_means[j]
                self.covs[rid] = new_covs[j]
                mapping[j] = rid
            return mapping

        template_ids = list(self.means.keys())
        cost = np.zeros((k_new, len(template_ids)))
        for j in range(k_new):
            for t_idx, rid in enumerate(template_ids):
                cost[j, t_idx] = wasserstein2_gaussian(
                    new_means[j], new_covs[j], self.means[rid], self.covs[rid]
                )
        row_ind, col_ind = linear_sum_assignment(cost)

        mapping: Dict[int, int] = {}
        matched_new = set()
        for j, t_idx in zip(row_ind, col_ind):
            rid = template_ids[t_idx]
            dist = cost[j, t_idx]
            thr = self.cfg.wasserstein_new_regime_threshold
            if thr is not None and dist > thr:
                new_rid = self.next_id
                self.next_id += 1
                self.means[new_rid] = new_means[j]
                self.covs[new_rid] = new_covs[j]
                mapping[j] = new_rid
            else:
                mapping[j] = rid
                self.means[rid] = (
                    (1 - self.ema_alpha) * self.means[rid] + self.ema_alpha * new_means[j]
                )
                self.covs[rid] = (
                    (1 - self.ema_alpha) * self.covs[rid] + self.ema_alpha * new_covs[j]
                )
            matched_new.add(j)

        for j in range(k_new):
            if j not in matched_new:
                rid = self.next_id
                self.next_id += 1
                self.means[rid] = new_means[j]
                self.covs[rid] = new_covs[j]
                mapping[j] = rid
        return mapping


# --------------------------------------------------------------------------- #
# RegimeCalibrator  (mirrors risk_engine.DailyCalibrator)
# --------------------------------------------------------------------------- #

class RegimeCalibrator:
    """
    Strictly causal, rolling Wasserstein-HMM regime calibrator.

    Fed the *factor-return* time series from an already-calibrated
    Barra-style risk model (functions_risk.get_factor_returns), not raw
    stock returns -- this is what ties regime identification to the
    cross-asset / factor level and lets it be mapped to individual
    stocks later via their factor exposures.
    """

    def __init__(self, cfg: RegimeModelConfig):
        self.cfg = cfg
        self.factor_returns_: Optional[pd.DataFrame] = None
        self.regime_prob_: Optional[pd.DataFrame] = None     # date x persistent_regime_id
        self.regime_label_: Optional[pd.Series] = None       # date -> persistent_regime_id
        self.transition_matrix_: Optional[pd.DataFrame] = None
        self.regime_mean_: Dict[int, pd.Series] = {}
        self.regime_cov_: Dict[int, pd.DataFrame] = {}
        self.model_order_history_: List[Tuple[pd.Timestamp, int]] = []
        self.current_regime_: Optional[int] = None
        self._template_book = _RegimeTemplateBook(cfg)
        self._last_model: Optional[_GaussianHMM] = None
        self._last_mapping: Optional[Dict[int, int]] = None

    # -- data ingestion -----------------------------------------------------
    def load_factor_returns(self, factor_returns: pd.DataFrame) -> None:
        self.factor_returns_ = factor_returns.sort_index().dropna(how="all")

    # -- walk-forward estimation --------------------------------------------
    def run(self, start=None, end=None) -> "RegimeCalibrator":
        if self.factor_returns_ is None:
            raise ValueError("Call load_factor_returns() before run().")

        fr = self.factor_returns_
        if start is not None:
            fr = fr[fr.index >= pd.Timestamp(start)]
        if end is not None:
            fr = fr[fr.index <= pd.Timestamp(end)]
        if len(fr) < self.cfg.estimation_window + self.cfg.order_selection_holdout:
            raise ValueError(
                f"Not enough factor-return history ({len(fr)} rows) for "
                f"estimation_window={self.cfg.estimation_window} + "
                f"order_selection_holdout={self.cfg.order_selection_holdout}."
            )

        d = fr.shape[1]
        usable_days = self.cfg.estimation_window - self.cfg.order_selection_holdout
        worst_case_params = n_hmm_params(d, self.cfg.max_regimes, self.cfg.cov_type)
        min_required = self.cfg.min_params_multiplier * worst_case_params
        if self.cfg.min_params_multiplier > 0 and usable_days < min_required:
            raise ValueError(
                f"estimation_window ({self.cfg.estimation_window}) - "
                f"order_selection_holdout ({self.cfg.order_selection_holdout}) = "
                f"{usable_days} training days is too few for a "
                f"{self.cfg.max_regimes}-state, {d}-dimensional ('{self.cfg.cov_type}' "
                f"covariance) HMM: {worst_case_params} free parameters, need >= "
                f"{self.cfg.min_params_multiplier:.0f}x = {min_required:.0f} training days. "
                f"Increase estimation_window, reduce max_regimes, use cov_type='diag', "
                f"or lower min_params_multiplier if you understand the risk."
            )

        # HMM fitting/labeling runs on causally-standardized inputs (B4);
        # regime_mean_/regime_cov_ below are computed on the RAW returns so
        # portfolio construction gets real expected-return units.
        fr_std = _causal_standardize(fr, self.cfg.standardize_halflife, self.cfg.standardize_min_periods)

        dates = fr.index
        prob_rows: Dict[pd.Timestamp, Dict[int, float]] = {}
        label_rows: Dict[pd.Timestamp, int] = {}

        refit_points = list(range(self.cfg.estimation_window, len(dates), self.cfg.refit_frequency))
        if refit_points[-1] != len(dates) - 1:
            refit_points.append(len(dates) - 1)

        window_start_idx = 0
        for end_idx in refit_points:
            start_idx = max(0, end_idx - self.cfg.estimation_window)
            X = fr_std.iloc[start_idx:end_idx + 1].values
            k, model, score = select_model_order(X, self.cfg)
            self.model_order_history_.append((dates[end_idx], k))

            mapping = self._template_book.match(model.means_, model.covs_)
            self._last_model, self._last_mapping = model, mapping

            filt = model.filtered_proba(X)
            # Only commit the NEW segment of days since the previous refit
            # (strictly causal -- earlier days keep their earlier-window
            # label). For the very FIRST window there is no earlier label
            # to keep: committing all estimation_window days would label
            # them with parameters fit on those same days (in-sample), so
            # only the last day is committed and the rest is burn-in.
            if end_idx == refit_points[0]:
                segment_start = end_idx - start_idx
            else:
                segment_start = window_start_idx - start_idx
            segment_dates = dates[start_idx + segment_start: end_idx + 1]
            segment_filt = filt[segment_start:]
            for dt, row in zip(segment_dates, segment_filt):
                probs = {mapping[j]: float(row[j]) for j in range(len(row))}
                prob_rows[dt] = probs
                label_rows[dt] = max(probs, key=probs.get)

            window_start_idx = end_idx + 1

        all_ids = sorted({rid for probs in prob_rows.values() for rid in probs})
        prob_df = pd.DataFrame(index=dates.intersection(list(prob_rows.keys())), columns=all_ids, dtype=float)
        for d, probs in prob_rows.items():
            for rid in all_ids:
                prob_df.loc[d, rid] = probs.get(rid, 0.0)
        prob_df = prob_df.sort_index().fillna(0.0)

        self.regime_prob_ = prob_df
        self.regime_label_ = pd.Series(label_rows).sort_index()

        # A component that the Wasserstein step could not match to a template
        # labels its days None, and the walk-forward can leave the tail
        # unlabelled altogether on a short history. Taking iloc[-1] blindly then
        # fails with "int() argument must be ... not 'NoneType'", which says
        # nothing about the cause.
        labelled = self.regime_label_.dropna()
        if labelled.empty:
            raise ValueError(
                "The walk-forward produced no regime labels: "
                f"{len(dates)} days with estimation_window="
                f"{self.cfg.estimation_window}, refit_frequency="
                f"{self.cfg.refit_frequency} left nothing to label. "
                "Use a longer factor history or a shorter estimation window."
            )
        self.current_regime_ = int(labelled.iloc[-1])

        # transition matrix + conditional moments in persistent-ID space,
        # taken from the most recent fit's mapping
        k_last = self._last_model.k
        inv_map = self._last_mapping
        trans_persistent = pd.DataFrame(0.0, index=all_ids, columns=all_ids)
        for i in range(k_last):
            for j in range(k_last):
                trans_persistent.loc[inv_map[i], inv_map[j]] = self._last_model.trans_[i, j]
        # renormalize rows touched
        row_sums = trans_persistent.sum(axis=1)
        for rid in all_ids:
            if row_sums.loc[rid] > 0:
                trans_persistent.loc[rid] = trans_persistent.loc[rid] / row_sums.loc[rid]
            elif rid == self.current_regime_:
                trans_persistent.loc[rid, rid] = 1.0
        self.transition_matrix_ = trans_persistent

        # Predictive conditional moments (B1 fix): weight the NEXT day's
        # raw return by the causal (filtered) belief at t, rather than
        # averaging a regime's own labeled days. The labels come from
        # fitting on those same returns, so an in-sample average of "days
        # labeled regime X" is biased high/low by construction -- it is not
        # an estimate of what regime X actually predicts.
        R_next = fr.shift(-1).loc[prob_df.index].dropna()
        P_aligned = prob_df.loc[R_next.index]
        for rid in all_ids:
            w = P_aligned[rid]
            w_sum = float(w.sum())
            if w_sum >= 5.0:
                mean = R_next.mul(w, axis=0).sum() / w_sum
                centered = R_next.sub(mean, axis=1)
                cov = (centered.mul(w, axis=0).T @ centered) / w_sum
                self.regime_mean_[rid] = mean
                self.regime_cov_[rid] = cov
            elif rid in self._template_book.means:
                cols = fr.columns
                self.regime_mean_[rid] = pd.Series(self._template_book.means[rid], index=cols)
                self.regime_cov_[rid] = pd.DataFrame(
                    self._template_book.covs[rid], index=cols, columns=cols
                )
            else:
                # Negligible belief-weighted mass and no template yet --
                # fall back to the unconditional sample as a last resort.
                self.regime_mean_[rid] = fr.mean()
                self.regime_cov_[rid] = fr.cov()

        return self

    # -- accessors ------------------------------------------------------
    def expected_dwell_time(self, regime: Optional[int] = None) -> float:
        """Expected sojourn time in trading days: 1 / (1 - p_ii)."""
        if regime is None:
            regime = self.current_regime_
        p_ii = float(self.transition_matrix_.loc[regime, regime])
        p_ii = min(p_ii, 0.999999)
        return 1.0 / (1.0 - p_ii)

    def summary(self) -> pd.DataFrame:
        rows = []
        for rid in self.transition_matrix_.index:
            days_active = int((self.regime_label_ == rid).sum())
            rows.append({
                "regime_id": rid,
                "is_current": rid == self.current_regime_,
                "persistence_p_ii": float(self.transition_matrix_.loc[rid, rid]),
                "expected_dwell_days": self.expected_dwell_time(rid),
                "days_observed": days_active,
                "pct_of_sample": 100.0 * days_active / len(self.regime_label_),
            })
        return pd.DataFrame(rows).set_index("regime_id").sort_values(
            "pct_of_sample", ascending=False
        )

    def report(self) -> str:
        cur = self.current_regime_
        dwell = self.expected_dwell_time()
        last_k = self.model_order_history_[-1][1] if self.model_order_history_ else None
        return (
            f"RegimeCalibrator: current_regime={cur}, "
            f"expected_dwell_days={dwell:.1f}, "
            f"active_model_order={last_k}, "
            f"regimes_tracked={len(self.transition_matrix_.index)}, "
            f"history_days={len(self.regime_label_)}"
        )

    def predicted_next_proba(self) -> pd.Series:
        """
        One-step-ahead predicted regime distribution pi_{T+1} = p_T . A,
        from the current filtered belief p_T and the persistent-ID
        transition matrix A. This -- not the hard argmax label at T, and
        not the filtered probability at T -- is the right notion of both
        "current regime confidence" and the input to a regime-conditional
        forecast, since it is the belief actually held about T+1 before
        T+1's return is known.
        """
        p_T = self.regime_prob_.iloc[-1]
        A = self.transition_matrix_.reindex(index=p_T.index, columns=p_T.index).fillna(0.0)
        pi_next = p_T.values @ A.values
        return pd.Series(pi_next, index=p_T.index)

    def conditional_moments(self, regime: Optional[int] = None) -> Tuple[pd.Series, pd.DataFrame]:
        """
        regime given: that persistent regime's predictive (belief-weighted,
        out-of-sample) mean/covariance -- e.g. for "what if we're in regime
        X" scenario analysis.

        regime=None: the ONE-STEP-AHEAD FORECAST, blending every regime's
        moments by the predicted next-day belief pi_{T+1} rather than
        hard-picking the current max-probability regime:
            mu_f = sum_k pi_{T+1,k} * mu_k
        This is what portfolio construction uses by default.
        """
        if regime is not None:
            return self.regime_mean_[regime], self.regime_cov_[regime]

        pi_next = self.predicted_next_proba()
        ids = [rid for rid in pi_next.index if rid in self.regime_mean_]
        w = pi_next.loc[ids]
        w = w / w.sum()
        mean_blend = sum(w[rid] * self.regime_mean_[rid] for rid in ids)
        cov_blend = sum(w[rid] * self.regime_cov_[rid] for rid in ids)
        return mean_blend, cov_blend


# --------------------------------------------------------------------------- #
# PortfolioConstructor
# --------------------------------------------------------------------------- #

class PortfolioConstructor:
    """
    Transaction-cost-aware mean-variance construction, conditioned on the
    current regime's factor moments, mapped to stocks via factor
    exposures (betas) from the Barra-style risk model.

    Objective (long-only, fully-invested):
        max_w   w'mu - (risk_aversion/2) w'Sigma w
                 - (tc_bps/1e4) * sum(|w - w_prev|)
        s.t.    sum(w) = 1, 0 <= w <= max_weight
    """

    def __init__(self, cfg: RegimeModelConfig):
        self.cfg = cfg

    def construct(
        self,
        exposures: pd.DataFrame,          # tickers x factors, latest cross-sectional betas
        factor_cov: pd.DataFrame,          # factors x factors, from the risk model (unconditional or regime-blended)
        idio_var: pd.Series,               # ticker -> idiosyncratic variance (daily)
        regime_factor_mean: pd.Series,     # factor -> regime-conditional expected return
        regime_id,
        regime_confidence: float,
        expected_hold_days: float,
        current_weights: Optional[pd.Series] = None,
        universe: Optional[List[str]] = None,
        target_n: Optional[int] = None,
        max_weight: Optional[float] = None,
        transaction_cost_bps: Optional[float] = None,
        risk_aversion: Optional[float] = None,
    ) -> pd.DataFrame:
        target_n = target_n or self.cfg.target_n
        max_weight = max_weight if max_weight is not None else self.cfg.max_weight
        tc_bps = transaction_cost_bps if transaction_cost_bps is not None else self.cfg.transaction_cost_bps
        gamma = risk_aversion if risk_aversion is not None else self.cfg.risk_aversion

        tickers = universe if universe is not None else list(exposures.index)
        tickers = [t for t in tickers if t in exposures.index]
        B = exposures.loc[tickers]
        factors = [f for f in B.columns if f in regime_factor_mean.index]
        B = B[factors]
        mu_f = regime_factor_mean[factors]
        Sigma_f = factor_cov.loc[factors, factors]

        mu = B.values @ mu_f.values                            # expected stock return, regime-conditional
        idio = idio_var.reindex(tickers).fillna(idio_var.median()).values
        Sigma = B.values @ Sigma_f.values @ B.values.T + np.diag(idio)

        mu_s = pd.Series(mu, index=tickers)

        if target_n is not None and target_n < len(tickers):
            info_ratio = mu_s / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
            tickers = list(info_ratio.sort_values(ascending=False).index[:target_n])
            idx = [B.index.get_loc(t) for t in tickers]
            mu = mu[idx]
            Sigma = Sigma[np.ix_(idx, idx)]

        n = len(tickers)
        w_prev = (
            current_weights.reindex(tickers).fillna(0.0).values
            if current_weights is not None
            else np.zeros(n)
        )

        # C1: optimise over the expected HOLDING PERIOD, not one day. mu and
        # Sigma both scale linearly with horizon so scaling them together
        # doesn't change the return/risk trade-off -- the actual distortion
        # was comparing a one-off transaction cost against a single day's
        # expected return while holding for `h` days. h is floored at 1 day.
        h = max(float(expected_hold_days), 1.0)
        mu_h, Sigma_h = mu * h, Sigma * h

        # C3: |w - w_prev| makes the SLSQP objective nonsmooth at w==w_prev,
        # which is exactly the low-turnover region this optimizer visits
        # most often. Split the trade into buys/sells b, s >= 0 with
        # w = w_prev + b - s: at any optimum of a convex objective with a
        # linear penalty on (b + s), the solver never has both b_i and s_i
        # positive simultaneously (that would only add cost), so
        # b_i + s_i == |w_i - w_prev_i| exactly -- same economics, smooth
        # objective, solvable by the same SLSQP call.
        lower = 0.0 if self.cfg.long_only else -max_weight

        def unpack(z):
            b, s = z[:n], z[n:]
            return w_prev + b - s

        def objective(z):
            w = unpack(z)
            ret = w @ mu_h
            risk = 0.5 * gamma * w @ Sigma_h @ w
            tc = (tc_bps / 1e4) * np.sum(z)
            return -(ret - risk - tc)

        cons = [
            {"type": "eq", "fun": lambda z: np.sum(unpack(z)) - 1.0},
            {"type": "ineq", "fun": lambda z: unpack(z) - lower},
            {"type": "ineq", "fun": lambda z: max_weight - unpack(z)},
        ]
        bounds = [(0.0, None)] * (2 * n)
        w0 = np.full(n, 1.0 / n)
        z0 = np.concatenate([np.maximum(w0 - w_prev, 0.0), np.maximum(w_prev - w0, 0.0)])
        result = minimize(objective, z0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
        if not result.success:
            raise RuntimeError(
                f"Portfolio optimization failed (status={result.status}): {result.message}"
            )
        w_opt = unpack(result.x)

        w_sum = float(np.sum(w_opt))
        if abs(w_sum - 1.0) > 1e-3:
            logger.warning(
                "Optimizer weights summed to %.6f (expected 1.0) -- solver reported "
                "success but constraints look violated; renormalizing.", w_sum,
            )
        w_opt = np.clip(w_opt, lower, max_weight)
        w_opt = w_opt / w_opt.sum()

        out = pd.DataFrame({
            "ticker": tickers,
            "weight": w_opt,
            "prev_weight": w_prev,
            "trade": w_opt - w_prev,
            "expected_return_regime": mu[:len(tickers)] if target_n is not None and target_n < len(exposures) else mu,
            "regime_id": regime_id,
            "regime_confidence": regime_confidence,
            "expected_hold_days": expected_hold_days,
        }).set_index("ticker")
        return out.sort_values("weight", ascending=False)

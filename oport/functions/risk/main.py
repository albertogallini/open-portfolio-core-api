"""
example_usage.py
================
End-to-end demonstration of the factor risk model.

Covers:
  1. Calibration over a historical window
  2. Inspecting factor returns and t-stats
  3. Factor covariance matrix
  4. Stock-level risk decomposition
  5. Portfolio risk decomposition
  6. Incremental daily update (live use-case)
"""
import datetime
import logging

import numpy as np
import pandas as pd

from risk_engine import ModelConfig, DailyCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── 0. Configuration ──────────────────────────────────────────────────────────

cfg = ModelConfig(
    # Tighten EWMA half-lives for a more reactive model
    factor_cov_halflife = 63,    # 3 months
    idio_var_halflife   = 21,    # 1 month
    wls_scheme          = "sqrt_mktcap",
    winsor_sigma        = 3.0,
    min_stocks_in_regression = 20, # Added this limit since our sample universe has 30 stocks, and default is 50
)

# ── 1. Calibration ────────────────────────────────────────────────────────────
# Provide your universe here.  Pass tickers=None to auto-load S&P 500.

MY_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM",
    "JNJ",  "V",    "PG",    "XOM",  "UNH",  "HD",   "MA",   "BAC",
    "ABBV", "PFE",  "AVGO",  "COST", "WMT",  "MRK",  "LLY",  "CVX",
    "ACN",  "ORCL", "TMO",   "MCD",  "CSCO", "ABT",
]

TRAIN_START = datetime.date(2022, 1, 1)
TRAIN_END   = datetime.date(2024, 12, 31)

cal = DailyCalibrator(cfg, tickers=None)
cal.load_data(start=TRAIN_START, end=TRAIN_END)
model = cal.run(start=TRAIN_START, end=TRAIN_END)

# ── 2. Print summary ─────────────────────────────────────────────────────────
cal.report()

# ── 3. Factor returns time series ────────────────────────────────────────────
fr = model.factor_returns
print("\n── Factor returns (last 5 days) ──")
print(fr.tail().round(4))

# Cumulative factor performance
cum_fr = (1 + fr).cumprod() - 1
print("\n── Cumulative factor returns ──")
print(cum_fr.iloc[-1].sort_values(ascending=False).round(4))

# ── 4. Factor t-stats (signal of factor significance) ────────────────────────
avg_t = model.factor_t_stats.mean()
pct_sig = (model.factor_t_stats.abs() > 2.0).mean() * 100
print("\n── Factor significance (avg |t|, % days |t| > 2) ──")
print(pd.DataFrame({"avg_|t|": avg_t.round(2), "pct_sig_%": pct_sig.round(1)}))

# ── 5. Factor covariance matrix ───────────────────────────────────────────────
F = model.get_factor_covariance()
print(f"\n── Factor covariance matrix  ({F.shape[0]} × {F.shape[1]}) ──")
# Convert to annualised correlation for readability
D_std = np.sqrt(np.diag(F.values) * 252)
corr  = F.values / np.outer(D_std, D_std)
print(pd.DataFrame(corr, index=F.index, columns=F.columns).round(2))

# ── 6. Stock-level risk decomposition ────────────────────────────────────────
decomp = cal.get_risk_decomposer()

# All stocks
risk_df = decomp.all_stocks_risk(scale=252)
print("\n── Stock risk decomposition (top 10 by total vol) ──")
print(risk_df.head(10).round(2))

# Single stock drill-down
for ticker in ["AAPL", "NVDA", "JPM"]:
    sr = decomp.stock_risk(ticker)
    if sr:
        print(
            f"  {ticker:6s}  total={sr.total_vol:5.1f}%  "
            f"factor={sr.factor_vol:5.1f}%  "
            f"idio={sr.idio_vol:5.1f}%  "
            f"factor_share={sr.factor_share:.1%}"
        )

# ── 7. Portfolio risk decomposition ──────────────────────────────────────────
# Equal-weight portfolio
ew_weights = pd.Series(
    1.0 / len(MY_UNIVERSE),
    index=MY_UNIVERSE,
)

pr = decomp.portfolio_risk(ew_weights)
print(f"\n── Equal-weight portfolio risk ──")
print(f"  Total vol      : {pr.total_vol:.2f}%")
print(f"  Factor vol     : {pr.factor_vol:.2f}%")
print(f"  Idio vol       : {pr.idio_vol:.2f}%")
print(f"  Factor share   : {pr.factor_share:.1%}")
print("\n  Per-factor variance contribution:")
print(pr.factor_contributions.sort_values(ascending=False).round(8).to_string())

# ── 8. Incremental daily update (live use-case) ───────────────────────────────
# After market close, just run one more day:
#
#   today = datetime.date.today()
#   cal.run(start=today, end=today)   ← fast, only one regression
#
# Then query updated risk:
#   decomp = cal.get_risk_decomposer()
#   print(decomp.stock_risk("AAPL"))

"""
factor_risk_model
=================
Barra-style cross-sectional factor risk model for US equities.

Quick start
-----------
>>> from factor_risk_model import ModelConfig, DailyCalibrator
>>> import datetime
>>>
>>> cfg = ModelConfig()
>>> cal = DailyCalibrator(cfg, tickers=my_universe)
>>> cal.load_data(start=datetime.date(2022, 1, 1), end=datetime.date(2024, 12, 31))
>>> model = cal.run(start=datetime.date(2023, 1, 1), end=datetime.date(2024, 12, 31))
>>>
>>> cal.report()
>>> decomp = cal.get_risk_decomposer()
>>> print(decomp.all_stocks_risk().head(20))
>>> print(decomp.portfolio_risk(equal_weight_portfolio))
"""
from .config     import ModelConfig
from .calibrator import DailyCalibrator
from .model      import (
    FactorRiskModel,
    RegressionResult,
    RiskDecomposer,
    StockRisk,
    PortfolioRisk,
    run_cross_sectional_regression,
    EWMACovarianceEstimator,
)
from .factors    import FactorBuilder
from .data       import (
    load_price_volume_panel,
    load_sp500_tickers,
    prefetch_sector_and_shares,
)

__all__ = [
    "ModelConfig",
    "DailyCalibrator",
    "FactorRiskModel",
    "RegressionResult",
    "RiskDecomposer",
    "StockRisk",
    "PortfolioRisk",
    "FactorBuilder",
    "EWMACovarianceEstimator",
    "run_cross_sectional_regression",
    "load_price_volume_panel",
    "load_sp500_tickers",
    "prefetch_sector_and_shares",
]

import unittest
import pandas as pd
import os
import shutil
from datetime import datetime
from oport.oport_api import (
    risk_calibration_logic,
    regime_calibration_logic,
    get_regime_summary_logic,
    construct_portfolio_logic,
    RISK_MODEL_FILE,
    REGIME_MODEL_FILE,
)
from oport import fields, error_collector


class TestRegimeApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Base data directory (relative path) -- same fixtures as test_risk_api.py
        cls.test_data_dir = "tests/test_attribution_md/test_portfolios"
        cls.universe_dir = os.path.join(cls.test_data_dir, "indexes")

        # Test targets
        cls.calibration_universe = "nasdaq_index"
        cls.portfolio_name = "test_e2e_stocksplits"

        # Calibration date range (matching stocksplits dataset part)
        cls.start_date = "2021-10-01"
        cls.end_date = "2021-11-05"
        cls.input_date = "2021-11-05"  # Friday
        cls.source = "filesystem"

        cls.risk_model_path = os.path.join(cls.universe_dir, RISK_MODEL_FILE)
        cls.regime_model_path = os.path.join(cls.universe_dir, REGIME_MODEL_FILE)
        cls.regime_model_path_portfolio = os.path.join(cls.test_data_dir, REGIME_MODEL_FILE)
        cls.risk_model_path_portfolio = os.path.join(cls.test_data_dir, RISK_MODEL_FILE)

        # regime_calibration_logic reads the risk model's own factor
        # returns, so it needs one already on disk. Calibrate it here
        # rather than depending on test_risk_api.py having run first --
        # keeps this test file self-contained.
        df_risk_summary = risk_calibration_logic(
            calibration_universe=cls.calibration_universe,
            start_date=cls.start_date,
            end_date=cls.end_date,
            source=cls.source,
            filesystem_folder=cls.universe_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )
        if df_risk_summary is None:
            collector = error_collector.get_collector()
            print("ERRORS found calibrating prerequisite risk model:", collector.get_errors())
        assert os.path.exists(cls.risk_model_path), (
            "Prerequisite risk model was not created -- regime tests cannot run."
        )

    def test_01_regime_calibration(self):
        print("\n=== Testing Regime Calibration ===")
        df_summary = regime_calibration_logic(
            source=self.source,
            filesystem_folder=self.universe_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
            start_date=self.start_date,
            end_date=self.end_date,
            estimation_window=15,
            order_selection_holdout=5,
            # This fixture has ~24 trading days total -- nowhere near enough
            # to satisfy the real min_params_multiplier guardrail (B5) for a
            # 9-dimensional HMM. Disabled here purely to keep this a fast
            # plumbing smoke test; see test_regime_engine.py for a synthetic
            # fixture sized to actually exercise the guardrail.
            min_params_multiplier=0.0,
        )
        if df_summary is None:
            collector = error_collector.get_collector()
            print("ERRORS found in regime calibration:", collector.get_errors())

        self.assertIsInstance(df_summary, pd.DataFrame)
        self.assertFalse(df_summary.empty, "Regime summary is empty")
        # summary is indexed by regime_id and should carry persistence /
        # dwell-time / sample-share columns -- see functions_regime.get_regime_summary
        for col in ("persistence_p_ii", "expected_dwell_days", "pct_of_sample"):
            self.assertIn(col, df_summary.columns)
        print("Regime calibration successful. Summary:")
        print(df_summary)
        self.assertTrue(os.path.exists(self.regime_model_path), "Serialized regime model not found")

    def test_02_regime_summary(self):
        print("\n=== Testing Regime Summary ===")
        df_current = get_regime_summary_logic(
            source=self.source,
            filesystem_folder=self.universe_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
        )
        self.assertIsInstance(df_current, pd.DataFrame)
        self.assertFalse(df_current.empty, "Current regime snapshot is empty")
        self.assertIn("confidence", df_current.columns)
        self.assertIn("expected_dwell_days", df_current.columns)
        # confidence is a filtered probability -- must be a valid probability
        conf = df_current["confidence"].iloc[0]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        print("Current regime snapshot:")
        print(df_current)

    def test_03_portfolio_construction(self):
        print("\n=== Testing Regime-Aware Portfolio Construction ===")
        # Same "everything in one filesystem_folder" constraint as
        # get_portfolio_risk_logic in test_risk_api.py: copy both models
        # into the portfolio's own folder alongside its holdings.
        for src, dst in (
            (self.risk_model_path, self.risk_model_path_portfolio),
            (self.regime_model_path, self.regime_model_path_portfolio),
        ):
            if os.path.exists(src):
                shutil.copy2(src, dst)

        df_portfolio = construct_portfolio_logic(
            universe_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=self.test_data_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
            input_date=self.input_date,
        )

        if df_portfolio is None:
            collector = error_collector.get_collector()
            print("ERRORS found in portfolio construction:", collector.get_errors())

        self.assertIsInstance(df_portfolio, pd.DataFrame)
        self.assertFalse(df_portfolio.empty, "Constructed portfolio is empty")
        for col in ("weight", "prev_weight", "trade", "regime_id",
                    "regime_confidence", "expected_hold_days"):
            self.assertIn(col, df_portfolio.columns)

        # weights should be non-negative (long-only default) and sum to ~1
        self.assertTrue((df_portfolio["weight"] >= -1e-9).all(), "Negative weight in long-only run")
        self.assertAlmostEqual(df_portfolio["weight"].sum(), 1.0, places=4)
        # every name should carry the same market-wide regime call
        self.assertEqual(df_portfolio["regime_id"].nunique(), 1)
        self.assertTrue((df_portfolio["expected_hold_days"] > 0).all())

        print("Constructed portfolio head:")
        print(df_portfolio.head())

    def test_04_portfolio_construction_target_n_and_max_weight(self):
        print("\n=== Testing Portfolio Construction with target_n / max_weight caps ===")
        target_n = 5
        max_weight = 0.30

        df_portfolio = construct_portfolio_logic(
            universe_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=self.test_data_dir,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
            input_date=self.input_date,
            target_n=target_n,
            max_weight=max_weight,
        )

        self.assertIsInstance(df_portfolio, pd.DataFrame)
        self.assertFalse(df_portfolio.empty)
        self.assertLessEqual(len(df_portfolio), target_n,
                              "Returned more names than the target_n cap")
        self.assertTrue((df_portfolio["weight"] <= max_weight + 1e-6).all(),
                         "A weight exceeded the max_weight cap")
        self.assertAlmostEqual(df_portfolio["weight"].sum(), 1.0, places=4)
        print(f"Portfolio capped at target_n={target_n}, max_weight={max_weight}:")
        print(df_portfolio)

    def test_05_portfolio_construction_missing_regime_model(self):
        print("\n=== Testing Portfolio Construction Failure Without a Regime Model ===")
        # Point at a folder that has a risk model but no regime model,
        # and confirm the logic fails gracefully (returns None + records
        # an error) rather than raising.
        empty_folder = os.path.join(self.test_data_dir, "no_regime_model_here")
        os.makedirs(empty_folder, exist_ok=True)
        if os.path.exists(self.risk_model_path):
            shutil.copy2(self.risk_model_path, os.path.join(empty_folder, RISK_MODEL_FILE))

        df_portfolio = construct_portfolio_logic(
            universe_name=self.portfolio_name,
            source=self.source,
            filesystem_folder=empty_folder,
            bucket_name=None,
            access_key_id=None,
            secret_access_key=None,
            s3_host=None,
            s3_port=None,
            ftp_host=None,
            ftp_user=None,
            ftp_pass=None,
            input_date=self.input_date,
        )

        self.assertIsNone(df_portfolio, "Expected None when no regime model is present")
        collector = error_collector.get_collector()
        errors = collector.get_errors()
        self.assertTrue(len(errors) > 0, "Expected an error to be recorded")

        shutil.rmtree(empty_folder, ignore_errors=True)

    @classmethod
    def tearDownClass(cls):
        # Cleanup
        # for path in [cls.risk_model_path, cls.regime_model_path,
        #              cls.risk_model_path_portfolio, cls.regime_model_path_portfolio]:
        #     if os.path.exists(path):
        #         os.remove(path)
        pass


if __name__ == "__main__":
    unittest.main()

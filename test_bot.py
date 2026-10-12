"""Regression tests for causality, costs, portfolio accounting and validation."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from Q_prog1 import (Config, PaperAccount, RegimeModel, engineer_features,
                     make_signals, normalize_prices, performance, run_pipeline,
                     simulate, synthetic_prices, target_exposure)


class BotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(train_end="2015-12-31", min_train=100, restarts=2)
        cls.prices, cls.diag = normalize_prices(synthetic_prices(periods=900), "^NSEI")
        cls.features = engineer_features(cls.prices, cls.cfg)
        cls.train = cls.features.loc[:cls.cfg.train_end]
        cls.model = RegimeModel().fit(cls.train, cls.cfg)
        cls.signals = cls.model.filter(cls.features)
        cls.signals["target_exposure"] = target_exposure(cls.signals, cls.features, cls.cfg)

    def test_forward_filter_prefix_invariance(self):
        prefix = self.features.iloc[:650]
        early = self.model.filter(prefix)
        full = self.model.filter(self.features).loc[prefix.index]
        np.testing.assert_allclose(early.prob_low_vol, full.prob_low_vol, atol=1e-12)

    def test_future_prices_do_not_change_past_features_or_signals(self):
        changed = self.prices.copy()
        boundary = changed.index[700]
        changed.loc[boundary:, ["Open", "High", "Low", "Close"]] *= 1.5
        new_features = engineer_features(changed, self.cfg)
        earlier = new_features.index < boundary
        np.testing.assert_allclose(new_features.loc[earlier, ["log_return", "vol_20"]],
                                   self.features.loc[new_features.index[earlier], ["log_return", "vol_20"]])
        new_signals = self.model.filter(new_features)
        np.testing.assert_allclose(new_signals.loc[earlier, "prob_low_vol"],
                                   self.signals.loc[new_features.index[earlier], "prob_low_vol"])

    def test_scaler_fitted_only_on_training(self):
        np.testing.assert_allclose(self.model.scaler.mean_, self.train[["log_return", "vol_20"]].mean())
        self.assertLess(self.model.train_end, self.features.index[-1])

    def test_transition_forecast_is_normalized(self):
        np.testing.assert_allclose(self.model.transition.sum(axis=1), 1)
        np.testing.assert_allclose(self.signals[["next_prob_low_vol", "next_prob_high_vol"]].sum(axis=1), 1)
        np.testing.assert_allclose(self.signals[["prob_low_vol", "prob_high_vol"]].sum(axis=1), 1)
        self.assertTrue(self.signals.transition_probability.between(0, 1).all())

    def test_exposure_varies_and_respects_volatility_cap(self):
        target = self.signals.target_exposure
        self.assertGreater(target.max() - target.min(), .1)
        self.assertTrue(target.between(0, 1).all())
        self.assertTrue((target <= self.cfg.target_vol / self.features.vol_20 + 1e-12).all())

    def test_rebalance_exact_cash_and_fee_accounting(self):
        account = PaperAccount(1000)
        notional, fee = account.rebalance(100, 1, .01)
        self.assertAlmostEqual(notional, 1000 / 1.01)
        self.assertAlmostEqual(account.cash, 0)
        self.assertAlmostEqual(account.value(100), 1000 - fee)
        wealth = account.value(110)
        _, fee = account.rebalance(110, .3, .01)
        self.assertAlmostEqual(account.value(110), wealth - fee)
        self.assertAlmostEqual(account.units * 110 / account.value(110), .3)
        wealth = account.value(110)
        _, fee = account.rebalance(110, 0, .01)
        self.assertAlmostEqual(account.units, 0)
        self.assertAlmostEqual(account.cash, wealth - fee)

    def test_next_open_execution_and_mark_to_market(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"Open": [100, 100, 120], "Close": [100, 110, 120]}, index=dates)
        signals = pd.DataFrame({"target_exposure": [1., 0., 1.], "confidence": 1.}, index=dates)
        cfg = Config(train_end="2020-01-01", initial_capital=1000,
                     cost_bps=0, slippage_bps=0, rebalance_band=0)
        ledger, trades = simulate(prices, signals, cfg)
        self.assertAlmostEqual(ledger.equity.iloc[0], 1100)
        self.assertAlmostEqual(ledger.equity.iloc[1], 1200)
        self.assertEqual(trades.side.tolist(), ["BUY", "SELL"])
        self.assertEqual(ledger.signal_date.iloc[0], dates[0])

    def test_signal_at_close_cannot_change_same_day_fill(self):
        baseline, _ = simulate(self.prices, self.signals, self.cfg)
        altered = self.signals.copy()
        date = baseline.index[10]
        altered.loc[date:, "target_exposure"] = 0
        other, _ = simulate(self.prices, altered, self.cfg)
        np.testing.assert_allclose(baseline.loc[:date, "equity"], other.loc[:date, "equity"])

    def test_entry_cost_and_initial_drawdown_are_counted(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"Open": 100., "Close": 90.}, index=dates)
        signals = pd.DataFrame({"target_exposure": 1., "confidence": 1.}, index=dates)
        cfg = Config(train_end="2020-01-01", initial_capital=1000, cost_bps=100, slippage_bps=0)
        ledger, _ = simulate(prices, signals, cfg, buy_hold=True)
        self.assertAlmostEqual(ledger.friction_paid.iloc[0], 1000 / 1.01 * .01)
        self.assertLess(performance(ledger, cfg)["max_drawdown"], -.1)

    def test_drawdown_stop_exits_next_open_and_stays_latched(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        prices = pd.DataFrame({"Open": [100., 100., 80., 90.], "Close": [100., 80., 90., 100.]}, index=dates)
        signals = pd.DataFrame({"target_exposure": 1., "confidence": 1.}, index=dates)
        cfg = Config(train_end="2020-01-01", initial_capital=1000, cost_bps=0,
                     slippage_bps=0, max_drawdown=.1)
        ledger, trades = simulate(prices, signals, cfg)
        self.assertTrue(ledger.stop_latched.all())
        self.assertEqual(ledger.target_at_open.tolist(), [1., 0., 0.])
        self.assertEqual(trades.side.tolist(), ["BUY", "SELL"])

    def test_walkforward_models_precede_each_test_signal(self):
        cfg = Config(train_end="2015-12-31", min_train=100, restarts=2, refit_every=200)
        signals, _, fits = make_signals(self.features, cfg)
        test = signals.loc[signals.index > pd.Timestamp(cfg.train_end)]
        self.assertGreater(len(fits), 1)
        self.assertTrue((test.model_train_end.to_numpy() < test.index.to_numpy()).all())

    def test_multiindex_and_flat_columns_are_equivalent(self):
        raw = self.prices.copy()
        raw.columns = pd.MultiIndex.from_product([raw.columns, ["^NSEI"]])
        flat, _ = normalize_prices(raw, "^NSEI")
        pd.testing.assert_frame_equal(flat, self.prices)

    def test_rejects_duplicate_dates_and_bad_prices(self):
        with self.assertRaises(ValueError):
            normalize_prices(pd.concat([self.prices, self.prices.iloc[:1]]), "^NSEI")
        bad = self.prices.copy()
        bad.iloc[0, bad.columns.get_loc("Close")] = -1
        with self.assertRaises(ValueError):
            normalize_prices(bad, "^NSEI")

    def test_simple_returns_match_price_ratios(self):
        expected = self.prices.Close.pct_change().loc[self.features.index]
        np.testing.assert_allclose(self.features.simple_return, expected, atol=1e-12)

    def test_full_pipeline_and_portable_saved_model(self):
        with tempfile.TemporaryDirectory() as temp:
            metrics, decision = run_pipeline(self.prices, self.cfg, Path(temp), "synthetic test", self.diag)
            self.assertEqual(set(metrics), {"regime_strategy", "buy_and_hold", "constant_30pct"})
            self.assertIn("Pending", decision["execution"])
            model = joblib.load(Path(temp) / "regime_model.joblib")
            self.assertIn("hmm", model)
            self.assertTrue((Path(temp) / "performance.png").is_file())
            ledger = pd.read_csv(Path(temp) / "regime_strategy_ledger.csv")
            self.assertTrue((ledger.cash >= 0).all())
            np.testing.assert_allclose((1 + ledger["return"]).cumprod() * self.cfg.initial_capital,
                                       ledger.equity, rtol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)

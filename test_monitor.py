"""Public Indian monitor tests; no market connection or brokerage required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from Q_prog1 import engineer_features
from indian_market_monitor import (analyze_bars, freshness, main, make_monitor_model,
                                   merge_bars, parse_args, prepare_bars)


def intraday_prices(sessions=27):
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2026-01-05", periods=sessions)
    index = pd.DatetimeIndex([date + pd.Timedelta(hours=9, minutes=15 + 5*k)
                              for date in days for k in range(75)], name="Date")
    state = np.repeat(np.arange(sessions) % 4 >= 2, 75)
    ret = rng.normal(.00001, np.where(state, .0018, .0004))
    close = 25000 * np.exp(np.cumsum(ret))
    opening = np.r_[25000., close[:-1]]
    return pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) * 1.0002,
                         "Low": np.minimum(opening, close) * .9998, "Close": close,
                         "Volume": 0}, index=index)


class MonitorTests(unittest.TestCase):
    def test_excludes_forming_bar_and_retains_distinct_intraday_times(self):
        raw = intraday_prices(1).iloc[:3]
        cleaned = prepare_bars(raw, "^NSEI", 5, "2026-01-05 09:26:00+05:30")
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned.index[-1].hour, 9)
        self.assertEqual(cleaned.index[-1].minute, 20)

    def test_utc_conversion_and_multiindex(self):
        raw = intraday_prices(1).iloc[:2]
        raw.index = raw.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
        raw.columns = pd.MultiIndex.from_product([raw.columns, ["^NSEI"]])
        cleaned = prepare_bars(raw, "^NSEI", 5, "2026-01-05 10:00:00+05:30")
        self.assertEqual(cleaned.index[0].hour, 9)
        self.assertEqual(cleaned.index[0].minute, 15)

    def test_fresh_stale_and_weekend_status_are_explicit(self):
        fresh = freshness("2026-01-05 09:20", 5, "2026-01-05 09:26+05:30")
        self.assertTrue(fresh["data_fresh_enough_for_monitor"])
        self.assertFalse(fresh["guaranteed_realtime"])
        stale = freshness("2026-01-05 09:20", 5, "2026-01-05 10:10+05:30")
        self.assertFalse(stale["data_fresh_enough_for_monitor"])
        weekend = freshness("2026-01-09 15:25", 5, "2026-01-10 10:00+05:30")
        self.assertEqual(weekend["status"], "OUTSIDE_REGULAR_SESSION")

    def test_cache_merge_overwrites_revisions_without_duplicate_bars(self):
        old = intraday_prices(1).iloc[:2]
        new = old.iloc[1:].copy()
        new.loc[:, "Close"] *= 1.01
        merged = merge_bars(old, new)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.Close.iloc[-1], new.Close.iloc[-1])

    def test_intraday_fit_saved_model_and_signal_pause(self):
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(["--output", temp])
            prices = intraday_prices()
            last = prices.index[-1]
            status = analyze_bars(prices, args, Path(temp), last + pd.Timedelta(minutes=6))
            self.assertEqual(status["freshness"]["status"], "OUTSIDE_REGULAR_SESSION")
            self.assertIsNone(status["current_research_target_exposure"])
            self.assertEqual(status["orders_submitted"], 0)
            original = pd.read_csv(Path(temp)/"intraday_features_and_signals.csv")
            # Reloading the trained artifact preserves filtered probabilities.
            again = analyze_bars(prices, args, Path(temp), last + pd.Timedelta(minutes=6))
            self.assertAlmostEqual(again["state_confidence"], status["state_confidence"])
            self.assertLess(pd.Timestamp(status["model_trained_through_ist"]).tz_localize(None), last)
            self.assertEqual(len(original), len(prices)-20)

    def test_feed_failure_never_produces_current_signal(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("indian_market_monitor.fetch_bars", side_effect=RuntimeError("provider unavailable")):
                self.assertEqual(main(["--once", "--output", temp]), 1)
            import json
            status = json.loads((Path(temp)/"latest_intraday_status.json").read_text())
            self.assertIsNone(status["current_research_target_exposure"])
            self.assertEqual(status["status"], "FEED_OR_MODEL_ERROR")

    def test_rejects_other_market_ticker(self):
        with self.assertRaises(SystemExit):
            parse_args(["--symbol", "AAPL"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

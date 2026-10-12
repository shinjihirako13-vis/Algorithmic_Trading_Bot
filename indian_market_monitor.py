#!/usr/bin/env python3
"""Broker-free Indian public intraday monitoring, NOT guaranteed real-time.

python Q_prog1.py --public-live --symbol NIFTY50 --poll 60
python indian_market_monitor.py --symbol BANKNIFTY --once

Uses public Yahoo intraday candles. No keys, broker account, orders or fills.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from Q_prog1 import Config, RegimeModel, engineer_features, normalize_prices, target_exposure, write_json

LOG = logging.getLogger("indian_monitor")
IST = "Asia/Kolkata"
SYMBOLS = {"NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
SOURCE = "Yahoo Finance public intraday candles via yfinance; delivery delay unverified"


def local_now(now=None):
    value = pd.Timestamp.now(tz=IST) if now is None else pd.Timestamp(now)
    return value.tz_localize(IST) if value.tz is None else value.tz_convert(IST)


def prepare_bars(raw, ticker, interval_minutes, now=None):
    """Keep complete regular-session bars, with timestamp interpreted in IST."""
    if raw is None or raw.empty:
        raise ValueError("Public intraday provider returned no data; it may be unavailable or rate-limited.")
    data = raw.copy()
    index = pd.to_datetime(data.index)
    if index.tz is None:
        # This mode is explicitly restricted to Indian instruments / IST inputs.
        index = index.tz_localize(IST)
    else:
        index = index.tz_convert(IST)
    data.index = index.tz_localize(None)
    minutes = data.index.hour * 60 + data.index.minute
    is_regular = (data.index.dayofweek < 5) & (minutes >= 555) & (minutes < 930)
    aligned = ((minutes - 555) % interval_minutes == 0) & (data.index.second == 0)
    if (is_regular & ~aligned).any():
        raise ValueError("Provider bar timestamps do not match the selected NSE/BSE interval grid.")
    instant = local_now(now).tz_localize(None)
    # Small publication grace, and reject the forming candle.
    completed = data.index + pd.Timedelta(minutes=interval_minutes, seconds=10) <= instant
    data = data.loc[is_regular & aligned & completed]
    if data.empty:
        raise ValueError("No completed regular-session intraday bars available.")
    return normalize_prices(data, ticker, daily=False)[0]


def freshness(last_start, interval_minutes, now=None, max_age_minutes=None):
    instant = local_now(now)
    end = pd.Timestamp(last_start).tz_localize(IST) + pd.Timedelta(minutes=interval_minutes)
    age = max(0., (instant - end).total_seconds() / 60)
    minute = instant.hour * 60 + instant.minute
    regular_window = instant.dayofweek < 5 and 555 <= minute < 930
    threshold = max_age_minutes if max_age_minutes is not None else 2 * interval_minutes + 2
    fresh = regular_window and end.date() == instant.date() and age <= threshold
    return {"checked_at_ist": instant.isoformat(), "latest_completed_bar_end_ist": end.isoformat(),
            "observed_bar_age_minutes": round(age, 3), "max_bar_age_minutes": threshold,
            "within_regular_weekday_session_window": bool(regular_window),
            "exchange_holiday_calendar_verified": False, "data_fresh_enough_for_monitor": bool(fresh),
            "status": "RECENT_PUBLIC_BAR" if fresh else ("STALE_OR_HOLIDAY" if regular_window else "OUTSIDE_REGULAR_SESSION"),
            "guaranteed_realtime": False,
            "note": "Bar age is observed candle-end age, not independently verified feed latency. Weekday hours do not certify exchange-open status."}


def fetch_bars(ticker, interval):
    import yfinance as yf
    # 59-day request stays inside the provider's documented intraday horizon.
    return yf.download(ticker, period="59d", interval=interval, auto_adjust=True,
                       progress=False, threads=False, timeout=15, ignore_tz=False)


def merge_bars(cached, incoming):
    # Revised provider rows replace cached rows; no duplicate timestamps persist.
    if cached is None:
        return incoming
    result = pd.concat([cached, incoming])
    return result.loc[~result.index.duplicated(keep="last")].sort_index()


def profile_for(args):
    return {"ticker": args.ticker, "interval": args.interval,
            "train_sessions": args.train_sessions, "seed": args.seed, "timezone": IST}


def make_monitor_model(features, args, output):
    artifact, metadata_path = output / "intraday_model.joblib", output / "monitor_profile.json"
    expected = profile_for(args)
    if metadata_path.exists():
        stored = json.loads(metadata_path.read_text())
        if stored != expected:
            raise ValueError("Output folder has a different ticker/interval/model profile; use another --output folder.")
    if artifact.exists() and not args.reset_model:
        # This is a local, generated artifact. Never load someone else's joblib.
        bundle = joblib.load(artifact)
        model = RegimeModel()
        model.hmm, model.scaler = bundle["hmm"], bundle["scaler"]
        model.order, model.train_end = bundle["order"], pd.Timestamp(bundle["train_end"])
        model.training_metadata = bundle["training_metadata"]
        cfg = Config(**bundle["config"])
        if features.index[0] > pd.Timestamp(bundle["context_start"]):
            raise ValueError("Historical filtering context is missing; restore the cache or use a new output folder.")
        return model, cfg
    sessions = features.index.normalize().unique()
    if len(sessions) <= args.train_sessions:
        raise ValueError(f"Need more than {args.train_sessions} sessions of intraday features; received {len(sessions)}.")
    train = features.loc[features.index.normalize().isin(sessions[:args.train_sessions])]
    minutes = int(args.interval[:-1])
    cfg = Config(start=str(features.index[0].date()), train_end=str(train.index[-1].date()),
                 annualization=252 * (375 // minutes), window=20,
                 min_train=252, restarts=3, seed=args.seed)
    cfg.validate()
    model = RegimeModel().fit(train, cfg)
    bundle = {"hmm": model.hmm, "scaler": model.scaler, "order": model.order,
              "train_end": model.train_end.isoformat(), "context_start": features.index[0].isoformat(),
              "training_metadata": model.training_metadata, "config": asdict(cfg)}
    temporary = artifact.with_suffix(".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(artifact)
    write_json(metadata_path, expected)
    return model, cfg


def analyze_bars(prices, args, output, now=None):
    # Rolling window is 20 bars in this mode. Annualization is intraday-specific.
    minutes = int(args.interval[:-1])
    feature_cfg = Config(annualization=252 * (375 // minutes))
    features = engineer_features(prices, feature_cfg)
    model, cfg = make_monitor_model(features, args, output)
    signals = model.filter(features)
    signals["target_exposure"] = target_exposure(signals, features, cfg)
    signals["sample"] = np.where(signals.index <= model.train_end, "train_descriptive", "after_training")
    latest = signals.iloc[-1]
    data_status = freshness(signals.index[-1], minutes, now, args.max_age_minutes)
    active = data_status["data_fresh_enough_for_monitor"] and signals.index[-1] > model.train_end
    payload = {"mode": "public Indian intraday research monitor; no brokerage or order execution",
               "symbol": args.symbol, "ticker": args.ticker, "source": SOURCE,
               "interval": args.interval, "freshness": data_status,
               "latest_completed_bar_start_ist": signals.index[-1].tz_localize(IST).isoformat(),
               "observed_close": float(features.Close.iloc[-1]),
               "model_trained_through_ist": model.train_end.tz_localize(IST).isoformat(),
               "regime": str(latest.regime_label), "state_confidence": float(latest.confidence),
               "next_bar_low_vol_probability": float(latest.next_prob_low_vol),
               "next_bar_high_vol_probability": float(latest.next_prob_high_vol),
               "state_change_probability": float(latest.transition_probability),
               "historical_model_target_exposure": float(latest.target_exposure),
               "current_research_target_exposure": float(latest.target_exposure) if active else None,
               "action": "RESEARCH_SIGNAL_ONLY" if active else "PAUSED_NO_CURRENT_SIGNAL",
               "orders_submitted": 0, "paper_fills_created": 0,
               "caution": "Polling completed candles is not tick streaming. Unknown provider delay means this must not be described as verified real-time data."}
    combined = features.join(signals)
    combined.to_csv(output / "intraday_features_and_signals.csv")
    write_json(output / "latest_intraday_status.json", payload)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="NIFTY50", help="NIFTY50, BANKNIFTY, SENSEX, or an NSE/BSE ticker such as RELIANCE.NS")
    parser.add_argument("--interval", choices=["5m", "15m"], default="5m")
    parser.add_argument("--poll", type=int, default=60, help="Seconds between public-data requests; minimum 60")
    parser.add_argument("--once", action="store_true", help="Fetch/analyze once and exit")
    parser.add_argument("--output", type=Path, default=Path("indian_monitor_output"))
    parser.add_argument("--train-sessions", type=int, default=20, help="Initial training-session count; model then stays fixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-age-minutes", type=float, help="Pause signal when last completed bar is older than this")
    parser.add_argument("--reset-model", action="store_true", help="Explicitly refit this profile once; does not remove the price cache")
    args = parser.parse_args(argv)
    args.symbol = args.symbol.upper()
    args.ticker = SYMBOLS.get(args.symbol, args.symbol)
    if args.ticker not in SYMBOLS.values() and not args.ticker.endswith((".NS", ".BO")):
        parser.error("Use NIFTY50/BANKNIFTY/SENSEX or an Indian .NS/.BO ticker.")
    if args.poll < 60 or args.train_sessions < 12:
        parser.error("--poll must be >=60; --train-sessions must be >=12.")
    if args.max_age_minutes is not None and (not np.isfinite(args.max_age_minutes) or args.max_age_minutes <= 0):
        parser.error("--max-age-minutes must be finite and positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    # Fail profile mismatches before reading/writing another instrument's cache.
    profile = output / "monitor_profile.json"
    if profile.exists() and json.loads(profile.read_text()) != profile_for(args):
        raise ValueError("Output profile mismatch; use a separate output folder for each instrument/interval.")
    cache_path = output / "intraday_prices_cache.csv"
    cached = pd.read_csv(cache_path, index_col="Date", parse_dates=True) if cache_path.exists() else None
    last_digest = None
    while True:
        try:
            incoming = prepare_bars(fetch_bars(args.ticker, args.interval), args.ticker, int(args.interval[:-1]))
            prices = merge_bars(cached, incoming)
            digest = hashlib.sha256(prices.to_csv().encode()).hexdigest()
            if digest != last_digest:
                status = analyze_bars(prices, args, output)
                prices.to_csv(cache_path)
                cached, last_digest = prices, digest
                args.reset_model = False
            else:
                status = json.loads((output / "latest_intraday_status.json").read_text())
                state = freshness(prices.index[-1], int(args.interval[:-1]), max_age_minutes=args.max_age_minutes)
                status["freshness"] = state
                status["current_research_target_exposure"] = status["historical_model_target_exposure"] if state["data_fresh_enough_for_monitor"] else None
                status["action"] = "RESEARCH_SIGNAL_ONLY" if state["data_fresh_enough_for_monitor"] else "PAUSED_NO_CURRENT_SIGNAL"
                write_json(output / "latest_intraday_status.json", status)
            print(json.dumps(status, indent=2), flush=True)
        except KeyboardInterrupt:
            LOG.info("Monitor stopped.")
            return 0
        except Exception as exc:
            last_digest = None
            error = {"mode": "public Indian intraday research monitor", "ticker": args.ticker,
                     "checked_at_ist": local_now().isoformat(), "status": "FEED_OR_MODEL_ERROR",
                     "error": str(exc), "guaranteed_realtime": False,
                     "current_research_target_exposure": None, "action": "PAUSED_NO_CURRENT_SIGNAL",
                     "orders_submitted": 0, "paper_fills_created": 0}
            write_json(output / "latest_intraday_status.json", error)
            LOG.error("%s", exc)
            if args.once:
                return 1
        if args.once:
            return 0
        try:
            time.sleep(args.poll)
        except KeyboardInterrupt:
            LOG.info("Monitor stopped.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())

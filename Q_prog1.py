#!/usr/bin/env python3
"""Daily HMM market-regime research and automated paper simulation.

Run: python Q_prog1.py --demo --output demo_output
     python Q_prog1.py --ticker '^NSEI' --train-end 2019-12-31
     python Q_prog1.py --csv prices.csv --refit-every 126

Signals use completed daily bars; paper rebalances execute at the next open.
No broker connection, real orders, leverage, or profit guarantee is included.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

LOG = logging.getLogger("regime_bot")
FEATURES = ["log_return", "vol_20"]


@dataclass(frozen=True)
class Config:
    ticker: str = "^NSEI"
    start: str = "2014-01-01"
    train_end: str = "2019-12-31"
    initial_capital: float = 100000.0
    annualization: int = 252
    window: int = 20
    min_train: int = 252
    restarts: int = 3
    n_iter: int = 500
    seed: int = 42
    refit_every: int = 0
    max_exposure: float = 1.0
    defensive_exposure: float = 0.3
    target_vol: float = 0.15
    min_confidence: float = 0.55
    rebalance_band: float = 0.02
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    max_drawdown: float = 0.25
    risk_free: float = 0.0

    def validate(self):
        if self.initial_capital <= 0 or self.annualization <= 0:
            raise ValueError("Capital and annualization must be positive.")
        if self.window < 2 or self.min_train < 50 or self.restarts < 1 or self.n_iter < 2:
            raise ValueError("Invalid window/training/model settings.")
        if self.refit_every < 0 or self.target_vol <= 0:
            raise ValueError("Refit interval must be nonnegative; target volatility positive.")
        if not 0 <= self.defensive_exposure <= self.max_exposure <= 1:
            raise ValueError("Require 0 <= defensive exposure <= max exposure <= 1.")
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.rebalance_band < 1:
            raise ValueError("Invalid confidence or rebalance band.")
        if not 0 < self.max_drawdown <= 1 or self.risk_free <= -1:
            raise ValueError("Invalid drawdown limit or risk-free assumption.")
        if min(self.cost_bps, self.slippage_bps) < 0 or self.friction >= 1:
            raise ValueError("Costs must be nonnegative and total friction below 100%.")
        if pd.Timestamp(self.start) >= pd.Timestamp(self.train_end):
            raise ValueError("Start must precede train-end.")

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10000.0


def write_json(path: Path, value):
    """Atomic JSON replacement so interrupted polling does not truncate state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def normalize_prices(raw: pd.DataFrame, ticker: str, *, daily=True):
    """Validate OHLC; preserve large moves. daily=False retains bar times."""
    if raw is None or raw.empty:
        raise ValueError("No prices received. Use --csv or --demo if download fails.")
    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        matches = [level for level in range(data.columns.nlevels)
                   if ticker in data.columns.get_level_values(level)]
        if matches:
            data = data.xs(ticker, level=matches[-1], axis=1)
        else:
            raise ValueError("Ticker not found in multi-level columns.")
    data.columns = [str(c).strip().title() for c in data.columns]
    if data.columns.duplicated().any():
        raise ValueError("Duplicate price-column names.")
    required = ["Open", "High", "Low", "Close"]
    if not set(required).issubset(data.columns):
        raise ValueError("Prices require Date, Open, High, Low, Close; Volume is optional.")
    index = pd.to_datetime(data.index, errors="coerce")
    if index.isna().any():
        raise ValueError("Invalid date in price input.")
    if index.tz is not None:
        index = index.tz_localize(None)
    data.index = index.normalize() if daily else index
    data.index.name = "Date"
    if data.index.duplicated().any():
        raise ValueError("Duplicate bar timestamps; resolve them rather than silently overwriting.")
    data = data.sort_index()
    cols = required + (["Volume"] if "Volume" in data else [])
    data = data[cols].apply(pd.to_numeric, errors="coerce")
    bad = (~np.isfinite(data[required]).all(axis=1)) | (data[required] <= 0).any(axis=1)
    bad |= data["High"] < data[["Open", "Close", "Low"]].max(axis=1)
    bad |= data["Low"] > data[["Open", "Close", "High"]].min(axis=1)
    if bad.any():
        raise ValueError(f"Invalid/missing OHLC on {int(bad.sum())} rows, e.g. {data.index[bad][0].date()}.")
    if "Volume" in data and ((data.Volume.dropna() < 0).any()):
        raise ValueError("Negative volume.")
    diagnostics = {
        "rows": len(data), "start": str(data.index[0].date()),
        "end": str(data.index[-1].date()), "duplicates": 0,
        "max_calendar_gap_days": int(data.index.to_series().diff().dt.days.max() or 0)
        if len(data) > 1 else 0,
        "missing_volume_rows": int(data.Volume.isna().sum()) if "Volume" in data else len(data),
        "policy": "Reject bad OHLC/duplicates; no forward fill, outlier clipping, or invented sessions.",
    }
    return data, diagnostics


def load_prices(args, cfg):
    if args.demo:
        return synthetic_prices(cfg.seed), "synthetic demonstration; NOT NIFTY performance"
    if args.csv:
        data = pd.read_csv(args.csv)
        date = next((c for c in data if c.lower().strip() == "date"), None)
        if date is None:
            raise ValueError("CSV must have a Date column.")
        return data.set_index(date), f"local CSV: {Path(args.csv).name}"
    import yfinance as yf
    # Exclusive end date avoids today's incomplete bar. Deliberately daily only.
    end = args.end or pd.Timestamp.now(tz=ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
    last_error = None
    for attempt in range(3):
        try:
            data = yf.download(cfg.ticker, start=cfg.start, end=end,
                               interval="1d", auto_adjust=True, progress=False,
                               threads=False, timeout=15)
            if data is not None and not data.empty:
                return data, "Yahoo Finance adjusted daily OHLC via yfinance"
            last_error = "empty download"
        except Exception as exc:
            last_error = str(exc)
        if attempt < 2:
            time.sleep(attempt + 1)
    raise RuntimeError(f"Market download failed: {last_error}. Supply --csv or --demo.")


def synthetic_prices(seed=42, periods=3200):
    """Two persistent synthetic volatility states; used only for smoke tests."""
    rng = np.random.default_rng(seed)
    states = np.zeros(periods, dtype=int)
    for t in range(1, periods):
        states[t] = (1 - states[t-1]) if rng.random() < 0.015 else states[t-1]
    daily = rng.normal(np.where(states == 0, 0.0004, -0.0002),
                       np.where(states == 0, 0.006, 0.022))
    close = 10000 * np.exp(np.cumsum(daily))
    previous = np.r_[10000., close[:-1]]
    opening = previous * np.exp(rng.normal(0, 0.002, periods))
    spread = rng.uniform(0.001, 0.009, periods)
    return pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) * (1 + spread),
                         "Low": np.minimum(opening, close) * (1 - spread), "Close": close,
                         "Volume": rng.integers(10000, 100000, periods)},
                        index=pd.bdate_range("2014-01-01", periods=periods, name="Date"))


def engineer_features(prices, cfg):
    features = prices.copy()
    features["log_return"] = np.log(features.Close / features.Close.shift(1))
    features["simple_return"] = np.expm1(features.log_return)
    features["vol_20"] = features.log_return.rolling(cfg.window).std(ddof=1) * np.sqrt(cfg.annualization)
    features = features.dropna(subset=FEATURES)
    if not np.isfinite(features[FEATURES].to_numpy()).all():
        raise ValueError("Nonfinite feature values.")
    if features.empty:
        raise ValueError("Insufficient history after rolling-window warmup.")
    return features


class RegimeModel:
    """Train-only scaling/full-covariance HMM and causal forward filtering.

    Emission geometry is a covariance-weighted (Mahalanobis) distance from
    each state's mean feature vector. This is Gaussian geometry, not an
    additional geometric prediction algorithm.
    """
    def fit(self, train, cfg):
        if len(train) < cfg.min_train:
            raise ValueError(f"Need at least {cfg.min_train} training feature rows; got {len(train)}.")
        if (train[FEATURES].std() < 1e-12).any():
            raise ValueError("Training features are constant or degenerate.")
        self.scaler = StandardScaler().fit(train[FEATURES])
        x = self.scaler.transform(train[FEATURES])
        best = None
        trials = []
        # Limit BLAS threads: tiny covariance matrices do not need large pools.
        with threadpool_limits(limits=1):
            for offset in range(cfg.restarts):
                model = GaussianHMM(n_components=2, covariance_type="full",
                                    n_iter=cfg.n_iter, tol=1e-3, min_covar=1e-4,
                                    random_state=cfg.seed + offset)
                try:
                    model.fit(x)
                    score = float(model.score(x))
                    hist = list(model.monitor_.history)
                    improvement = float(hist[-1] - hist[-2]) if len(hist) > 1 else math.inf
                    stable = np.isfinite(improvement) and -1e-3 <= improvement < model.tol
                    posterior_mass = model.predict_proba(x).mean(axis=0)
                    usable = stable and np.isfinite(score) and posterior_mass.min() > 0.01
                    trials.append({"seed": cfg.seed + offset, "log_likelihood": score,
                                   "iterations": int(model.monitor_.iter), "stable": bool(stable),
                                   "minimum_state_mass": float(posterior_mass.min())})
                    if usable and (best is None or score > best[0]):
                        best = (score, model)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    trials.append({"seed": cfg.seed + offset, "error": str(exc)})
        if best is None:
            raise RuntimeError("No stable, noncollapsed HMM fit. Increase --n-iter/--restarts or inspect training data.")
        self.hmm = best[1]
        raw_means = self.scaler.inverse_transform(self.hmm.means_)
        self.order = np.argsort(raw_means[:, FEATURES.index("vol_20")])
        self.train_end = train.index[-1]
        self.training_metadata = {
            "train_start": str(train.index[0].date()), "train_end": str(self.train_end.date()),
            "training_rows": len(train), "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(), "trials": trials,
            "state_order_low_to_high_vol": self.order.tolist(),
            "state_mean_features": raw_means[self.order].tolist(),
            "transition_matrix": self.transition.tolist(),
        }
        return self

    @property
    def transition(self):
        return self.hmm.transmat_[np.ix_(self.order, self.order)]

    def filter(self, features):
        """P(state_t | observations through t), never Viterbi/smoothed test labels."""
        x = self.scaler.transform(features[FEATURES])
        emission = np.column_stack([
            multivariate_normal.logpdf(x, mean=mean, cov=cov)
            for mean, cov in zip(self.hmm.means_, self.hmm.covars_)
        ])
        with np.errstate(divide="ignore"):
            log_transition = np.log(self.hmm.transmat_)
            prior = np.log(self.hmm.startprob_)
        probabilities = np.empty((len(x), 2))
        for t, likelihood in enumerate(emission):
            if t:
                prior = logsumexp(posterior[:, None] + log_transition, axis=0)
            posterior = prior + likelihood
            posterior -= logsumexp(posterior)
            probabilities[t] = np.exp(posterior)
        ordered = probabilities[:, self.order]
        forecast = ordered @ self.transition
        distance = np.column_stack([
            np.einsum("ij,jk,ik->i", x - self.hmm.means_[k],
                      np.linalg.inv(self.hmm.covars_[k]), x - self.hmm.means_[k])
            for k in self.order
        ])
        result = pd.DataFrame(index=features.index)
        result[["prob_low_vol", "prob_high_vol"]] = ordered
        result[["next_prob_low_vol", "next_prob_high_vol"]] = forecast
        result[["mahalanobis2_low_vol", "mahalanobis2_high_vol"]] = distance
        result["regime"] = ordered.argmax(axis=1)
        result["regime_label"] = np.where(result.regime == 0, "Low-Vol", "High-Vol")
        result["confidence"] = ordered.max(axis=1)
        result["transition_probability"] = ordered[:, 0] * self.transition[0, 1] + ordered[:, 1] * self.transition[1, 0]
        result["model_train_end"] = self.train_end
        return result


def target_exposure(signals, features, cfg):
    # Probability-weighted exposure: labels cannot cause string mismatch bugs.
    base = cfg.defensive_exposure + (cfg.max_exposure - cfg.defensive_exposure) * signals.next_prob_low_vol
    base = base.where(signals.confidence >= cfg.min_confidence, cfg.defensive_exposure)
    vol_cap = cfg.target_vol / features.vol_20.clip(lower=1e-8)
    return np.minimum(base, vol_cap).clip(0, cfg.max_exposure)


def make_signals(features, cfg):
    cutoff = pd.Timestamp(cfg.train_end)
    train = features.loc[features.index <= cutoff]
    test = features.loc[features.index > cutoff]
    if len(test) < 2:
        raise ValueError("Need at least two completed test sessions after train-end.")
    model = RegimeModel().fit(train, cfg)
    metadata = [model.training_metadata]
    # Historical training labels are descriptive only; backtest excludes them.
    pieces = [model.filter(train)]
    size = cfg.refit_every or len(test)
    for start in range(0, len(test), size):
        block = test.iloc[start:start + size]
        if start:
            prior = features.loc[features.index < block.index[0]]
            model = RegimeModel().fit(prior, cfg)
            metadata.append(model.training_metadata)
        history = features.loc[features.index <= block.index[-1]]
        pieces.append(model.filter(history).loc[block.index])
    signals = pd.concat(pieces).sort_index()
    signals["target_exposure"] = target_exposure(signals, features, cfg)
    signals["sample"] = np.where(signals.index <= cutoff, "train_descriptive", "out_of_sample")
    return signals, model, metadata


@dataclass
class PaperAccount:
    """Fractional proxy units; dollar fees include assumed slippage."""
    cash: float
    units: float = 0.0

    def value(self, price):
        return self.cash + self.units * price

    def rebalance(self, price, target, friction):
        wealth = self.value(price)
        risky = self.units * price
        gap = target * wealth - risky
        sign = 1.0 if gap >= 0 else -1.0
        # Solve risky_after = target * (wealth_before - fee) exactly.
        notional = gap / (1 + sign * target * friction)
        fee = abs(notional) * friction
        self.units += notional / price
        self.cash -= notional + fee
        if self.cash < -1e-6 or self.units < -1e-9:
            raise ArithmeticError("Paper account violated no-leverage/no-short constraints.")
        self.cash = max(self.cash, 0.0)
        return notional, fee


def simulate(prices, signals, cfg, *, buy_hold=False, fixed_exposure=None):
    """Open fills from previous close signals; daily close mark-to-market.

    Includes first entry and drift-driven turnover. No forced last-day exit:
    results and paper account are marked at the last available close.
    """
    previous = signals.target_exposure.shift(1)
    signal_date = pd.Series(signals.index, index=signals.index).shift(1)
    previous_confidence = signals.confidence.shift(1)
    dates = signals.index[signals.index > pd.Timestamp(cfg.train_end)]
    account = PaperAccount(cfg.initial_capital)
    peak = cfg.initial_capital
    close_peak = cfg.initial_capital
    stopped = False
    last_close_equity = cfg.initial_capital
    rows, trades = [], []
    for date in dates:
        opening, close = float(prices.at[date, "Open"]), float(prices.at[date, "Close"])
        pretrade = account.value(opening)
        # Observe only current open and previous marks; exit at this open.
        peak = max(peak, pretrade)
        if not buy_hold and fixed_exposure is None and pretrade / peak - 1 <= -cfg.max_drawdown:
            stopped = True
        if buy_hold:
            target = 1.0
        elif fixed_exposure is not None:
            target = fixed_exposure
        else:
            target = 0.0 if stopped else float(previous.at[date])
        actual = account.units * opening / pretrade
        force = buy_hold and not rows
        execute = force or (not buy_hold and abs(target - actual) >= cfg.rebalance_band)
        # Always honour cash exit, even within the rebalance band.
        execute |= not buy_hold and target == 0 and account.units > 0
        notional, fee = account.rebalance(opening, target, cfg.friction) if execute else (0., 0.)
        if abs(notional) > 1e-7:
            trades.append({"Date": str(date.date()), "signal_date": str(signal_date.at[date].date()),
                           "side": "BUY" if notional > 0 else "SELL", "reference_open": opening,
                           "units": abs(notional) / opening, "signed_notional": notional,
                           "friction_paid": fee, "target_exposure": target,
                           "reason": "drawdown_stop" if stopped else "rebalance"})
        equity = account.value(close)
        daily_return = equity / last_close_equity - 1
        peak = max(peak, equity)
        close_peak = max(close_peak, equity)
        # Close breach schedules liquidation at next session open.
        if not buy_hold and fixed_exposure is None and equity / peak - 1 <= -cfg.max_drawdown:
            stopped = True
        rows.append({"Date": date, "equity": equity, "return": daily_return,
                     "cash": account.cash, "units": account.units,
                     "exposure": account.units * close / equity, "target_at_open": target,
                     "signal_date": signal_date.at[date],
                     "signal_confidence": float(previous_confidence.at[date]),
                     "turnover": abs(notional) / pretrade, "friction_paid": fee,
                     "drawdown": equity / close_peak - 1, "stop_latched": stopped})
        last_close_equity = equity
    frame = pd.DataFrame(rows).set_index("Date")
    columns = ["Date", "signal_date", "side", "reference_open", "units", "signed_notional",
               "friction_paid", "target_exposure", "reason"]
    return frame, pd.DataFrame(trades, columns=columns)


def performance(frame, cfg):
    returns = frame["return"]
    sigma = float(returns.std(ddof=1))
    daily_rf = (1 + cfg.risk_free) ** (1 / cfg.annualization) - 1
    ratio = float((returns.mean() - daily_rf) / sigma * np.sqrt(cfg.annualization)) if sigma > 1e-12 else None
    total = float(frame.equity.iloc[-1] / cfg.initial_capital - 1)
    return {"sessions": len(frame), "total_return": total,
            "annualized_return_252_sessions": float((1 + total) ** (cfg.annualization / len(frame)) - 1),
            "sharpe": ratio, "annualized_volatility": sigma * np.sqrt(cfg.annualization),
            "max_drawdown": float(frame.drawdown.min()),
            "total_friction_paid": float(frame.friction_paid.sum()),
            "total_turnover": float(frame.turnover.sum()),
            "average_close_exposure": float(frame.exposure.mean()),
            "rebalance_count": int((frame.turnover > 1e-12).sum()),
            "terminal_equity": float(frame.equity.iloc[-1])}


def save_plots(features, signals, runs, output, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    fig.suptitle("Source: " + source, fontsize=12)
    axes[0].plot(features.index, features.Close, color="grey", alpha=.5)
    for state, color in [(0, "tab:blue"), (1, "tab:red")]:
        idx = signals.index[signals.regime == state]
        axes[0].scatter(idx, features.loc[idx, "Close"], s=3, color=color,
                        label="Low volatility" if state == 0 else "High volatility")
    axes[0].set_title("Historical regime inference (training segment is descriptive)")
    for name, frame in runs.items():
        axes[1].plot(frame.index, frame.equity, label=name)
        axes[2].plot(frame.index, frame.drawdown, label=name)
    axes[1].set_title("Out-of-sample paper equity after assumed costs")
    axes[2].set_title("Drawdown from initial capital / running equity peak")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(output / "performance.png", dpi=160)
    plt.close(fig)


def run_pipeline(prices, cfg, output, source, diagnostics):
    output.mkdir(parents=True, exist_ok=True)
    features = engineer_features(prices, cfg)
    signals, model, metadata = make_signals(features, cfg)
    runs, trade_logs = {}, {}
    for name, options in [("regime_strategy", {}), ("buy_and_hold", {"buy_hold": True}),
                          ("constant_30pct", {"fixed_exposure": 0.3})]:
        runs[name], trade_logs[name] = simulate(prices, signals, cfg, **options)
    metrics = {name: performance(frame, cfg) for name, frame in runs.items()}
    latest = signals.iloc[-1]
    account = runs["regime_strategy"].iloc[-1]
    close = float(prices.Close.iloc[-1])
    next_target = 0.0 if bool(account.stop_latched) else float(latest.target_exposure)
    illustrative = PaperAccount(float(account.cash), float(account.units))
    notional, fee = illustrative.rebalance(close, next_target, cfg.friction)
    if abs(next_target - account.exposure) < cfg.rebalance_band and not (next_target == 0 and account.units > 0):
        notional, fee = 0., 0.
    decision = {
        "mode": "paper simulation only", "source": source,
        "as_of_completed_session": str(signals.index[-1].date()),
        "model_trained_through": str(pd.Timestamp(latest.model_train_end).date()),
        "regime": str(latest.regime_label), "confidence": float(latest.confidence),
        "next_session_low_vol_probability": float(latest.next_prob_low_vol),
        "next_session_high_vol_probability": float(latest.next_prob_high_vol),
        "probability_of_state_change": float(latest.transition_probability),
        "next_open_target_exposure": next_target, "drawdown_stop_latched": bool(account.stop_latched),
        "paper_account": {"cash": float(account.cash), "proxy_units": float(account.units),
                          "marked_equity": float(account.equity)},
        "illustrative_order": {"side": "HOLD" if abs(notional) < 1e-7 else ("BUY" if notional > 0 else "SELL"),
                               "signed_notional_at_last_close": notional, "estimated_friction": fee},
        "execution": "Pending next observed open; recalculate quantity at open. This is NOT a fill or broker order.",
    }
    prices.to_csv(output / "prices_clean.csv")
    features.join(signals).to_csv(output / "features_and_signals.csv")
    for name, frame in runs.items():
        frame.to_csv(output / f"{name}_ledger.csv")
        trade_logs[name].to_csv(output / f"{name}_trades.csv", index=False)
    train_idx = signals.index <= pd.Timestamp(cfg.train_end)
    stats = features.loc[train_idx, FEATURES].join(signals.loc[train_idx, "regime_label"]).groupby("regime_label").agg(
        mean_log_return=("log_return", "mean"), return_std=("log_return", "std"),
        mean_annualized_vol=("vol_20", "mean"), sessions=("log_return", "size"))
    stats.to_csv(output / "training_regime_statistics.csv")
    pd.DataFrame(model.transition, index=["Low-Vol", "High-Vol"],
                 columns=["Low-Vol", "High-Vol"]).to_csv(output / "latest_transition_matrix.csv")
    joblib.dump({"hmm": model.hmm, "scaler": model.scaler, "state_order": model.order,
                 "features": FEATURES, "training_metadata": model.training_metadata},
                output / "regime_model.joblib")
    versions = {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "joblib", "matplotlib"]}
    write_json(output / "run_metadata.json", {
        "config": asdict(cfg), "source": source, "data_validation": diagnostics,
        "price_sha256": hashlib.sha256(prices.to_csv().encode()).hexdigest(),
        "versions": versions, "model_fits": metadata,
        "assumptions": ["Daily completed OHLC, next-open paper execution; fractional proxy units.",
                        "Commission and slippage charged on traded notional, both sides; cash earns zero.",
                        "No dividends beyond adjusted input, taxes, liquidity/impact, or broker fills modeled.",
                        "Index is a research proxy, not directly tradable; use a tradable instrument for implementation.",
                        "Fixed train/test or expanding-window refits; no holdout-based parameter selection.",
                        "Drawdown stop is permanent within this run and cannot prevent overnight gap losses.",
                        "Watch rebuilds a deterministic historical paper ledger, not a broker account."]})
    write_json(output / "metrics.json", metrics)
    write_json(output / "latest_paper_decision.json", decision)
    save_plots(features, signals, runs, output, source)
    LOG.info("Saved %s; latest %s, next target %.1f%%", output,
             decision["as_of_completed_session"], next_target * 100)
    return metrics, decision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--csv", type=Path, help="Completed daily OHLC CSV with Date column")
    inputs.add_argument("--demo", action="store_true", help="Synthetic data, never actual market results")
    parser.add_argument("--public-live", action="store_true", help="Broker-free Indian intraday monitor; use --public-live --help for options")
    parser.add_argument("--ticker", default="^NSEI")
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", help="Exclusive download end; default today's India date")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--output", type=Path, default=Path("bot_output"))
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="Poll completed daily data; requires >=60")
    parser.add_argument("--refit-every", type=int, default=0, help="Expanding-window refit every N test sessions; 0=fixed model")
    parser.add_argument("--initial-capital", type=float, default=100000)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-bps", type=float, default=10)
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--max-exposure", type=float, default=1)
    parser.add_argument("--defensive-exposure", type=float, default=.3)
    parser.add_argument("--target-vol", type=float, default=.15)
    parser.add_argument("--min-confidence", type=float, default=.55)
    parser.add_argument("--rebalance-band", type=float, default=.02)
    parser.add_argument("--max-drawdown", type=float, default=.25)
    parser.add_argument("--risk-free", type=float, default=0)
    args = parser.parse_args(argv)
    if args.watch and (args.watch < 60 or args.demo or args.end):
        parser.error("--watch requires >=60 seconds, non-demo data, and no fixed --end.")
    return args


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--public-live" in arguments:
        arguments.remove("--public-live")
        from indian_market_monitor import main as monitor_main
        return monitor_main(arguments)
    args = parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(**{key: getattr(args, key) for key in asdict(Config()) if hasattr(args, key)})
    cfg.validate()
    last_digest = None
    while True:
        try:
            raw, source = load_prices(args, cfg)
            prices, diagnostics = normalize_prices(raw, cfg.ticker)
            prices = prices.loc[prices.index >= pd.Timestamp(cfg.start)]
            if prices.empty:
                raise ValueError("No prices on or after --start.")
            digest = hashlib.sha256(prices.to_csv().encode()).hexdigest()
            if digest != last_digest:
                metrics, decision = run_pipeline(prices, cfg, args.output, source, diagnostics)
                print(json.dumps({"metrics": metrics, "latest_paper_decision": decision}, indent=2))
                last_digest = digest
            else:
                LOG.info("No new completed daily data; no duplicate paper trades.")
        except Exception:
            if not args.watch:
                raise
            LOG.exception("Polling failed; retaining last successful reports and retrying.")
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.info("Stopped.")
    except Exception as exc:
        LOG.error("%s", exc)
        raise SystemExit(1) from exc

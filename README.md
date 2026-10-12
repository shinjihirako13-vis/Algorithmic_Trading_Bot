# Algorithmic Trading and Market Regime Detection Bot

A modular Python implementation of the supplied project objectives, with daily research plus public Indian intraday monitoring,
regime inference, next-regime probability forecasts, a realistic paper-account
simulation, and repeatable reports. It replaces the original notebook-export
script. The modules are classes/functions in one importable script so it remains
easy to run and extend.

**Scope:** research and automated historical paper simulation. There is no broker
connection or live-money order execution. NIFTY 50 is an index and is used as a
fractional research proxy, not as an executable instrument. To implement actual
trading, select a tradable instrument and add its broker API, instrument-specific
lot sizes, exchange calendar, order lifecycle, and reconciliation.

## Indian intraday mode without a broker

Read INDIAN_MARKET_SETUP.md for the public Indian-market monitor. Start with:

```bash
python Q_prog1.py --public-live --symbol NIFTY50 --poll 60
```

This uses public five-minute candles with unverified delivery delay, not a
guaranteed real-time exchange feed. It requires no broker account and submits
no orders. The monitor has its own intraday-trained model and stale-data checks.
Actual Yahoo access was rate-limited during validation.

## Install and run

Use Python 3.11 or newer (validated here with Python 3.12).

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_bot test_monitor
python Q_prog1.py --demo --output demo_output
```

`--demo` generates synthetic prices and does not contact Yahoo Finance. All demo
results must be described as synthetic, never as NIFTY or real trading results.

Download completed daily NIFTY bars:

```bash
python Q_prog1.py --ticker "^NSEI" --train-end 2019-12-31 --output nifty_output
```

For reproducible runs use an exclusive end date:

```bash
python Q_prog1.py --ticker "^NSEI" --end 2026-10-12 --output nifty_output
```

Yahoo Finance may rate-limit or return no data. No synthetic fallback is silently
substituted. In that case use your own OHLC CSV:

```bash
python Q_prog1.py --csv prices.csv --train-end 2019-12-31 --output csv_output
```

CSV format (these rows illustrate the schema, not sufficient training history):

```csv
Date,Open,High,Low,Close,Volume
2014-01-02,6300,6340,6250,6301,100000
2014-01-03,6305,6380,6290,6360,110000
```

Supply consistently adjusted prices with one completed daily bar per session.
Volume is optional and is not a model feature. Do not supply incomplete daily bars,
mixed adjusted/unadjusted OHLC, or intraday data. A daily trading calendar is not
inferred from weekdays: holidays are retained as natural gaps. Gaps are reported,
but the program cannot identify missing exchange sessions without a calendar.

The default training requirement is 252 feature rows, plus 20 warmup price rows.
At least two completed sessions must follow `--train-end`.

Expanding-window evaluation refits using only history before each next block:

```bash
python Q_prog1.py --csv prices.csv --refit-every 126 --output walkforward_output
```

Poll daily data while your terminal stays open:

```bash
python Q_prog1.py --ticker "^NSEI" --watch 300 --output paper_output
```

Stop with Ctrl+C. This refreshes reports when completed prices change, rebuilds
the deterministic paper ledger, and updates the next-open decision. Unchanged
input does not produce another trade. It is not a running broker account or an
intraday price feed. The default download excludes today's India date to avoid
partial bars, so the newest session becomes available on a subsequent day. For
other markets, use a CSV or explicitly select a completed-data end date in a
one-off run. The scheduled retraining parameter applies within replay; it does
not create a system scheduler.

## Objective coverage

| Objective | Implementation |
|---|---|
| Intelligent automated framework | Download/CSV ingestion, regime inference, probability-weighted exposure, risk controls, daily polling and pending paper decisions |
| Preprocessing and normalization | OHLC/date validation, MultiIndex handling, chronological sorting, duplicate rejection, rolling features and train-only StandardScaler |
| HMM, statistical learning and vector geometry | Full-covariance two-state Gaussian HMM, multiple training seeds, convergence/occupancy checks, forward-filtered state probabilities, next-state probabilities, Mahalanobis distances |
| Modular backtesting and metrics | Fractional paper account, next-open fills, fees and slippage, portfolio ledgers, drawdown stop, Sharpe/volatility/drawdown/annualized return/turnover, benchmark comparisons |
| Technology stack | Python, NumPy, Pandas, Scikit-learn, hmmlearn, SciPy, Matplotlib, joblib and yfinance |

## How it works

1. Validate and sort prices; reject invalid OHLC, bad dates and duplicate sessions.
   Large price moves are not clipped, and missing prices are not forward-filled.
2. Calculate log return and trailing 20-session annualized volatility. The name
   `vol_20` refers to the default window in Config.
3. Fit StandardScaler on training features only. Fit a two-state Gaussian HMM with
   several seeds, retaining the highest training likelihood among stable fits
   with noncollapsed state occupancy. The test set never selects a seed or fits
   the initial normalization.
4. Label states by their fitted mean rolling volatility: Low-Vol and High-Vol.
   Low volatility is not assumed to guarantee positive returns.
5. Infer probabilities through forward recursion:

   `P(state_t | data through t)` is proportional to the state's emission likelihood
   multiplied by the transition-weighted previous filtered probability.

   It does not use full-sequence Viterbi or smoothed test probabilities. Training
   labels are descriptive because fitted training parameters use the full training
   segment. Only post-cutoff rows are evaluated as out-of-sample returns.
6. Forecast the next state distribution with `filtered_probability @ transition`.
   The state-change forecast is `p_low*T[low,high] + p_high*T[high,low]`. This
   predicts a distribution, not a guaranteed market event or a price direction.
7. Use covariance-weighted vector distances for Gaussian state geometry:
   `(x - mean).T @ inverse(covariance) @ (x - mean)`. These distances are diagnostics
   underlying the Gaussian model, not a separate geometric trading predictor.
8. Construct an exposure target between 30% and 100% from the next low-volatility
   probability. Low-confidence observations use the defensive target. Cap exposure
   by target annualized volatility divided by observed trailing volatility.
9. Execute yesterday's close signal at today's observed open in the paper model.
   Carry existing holdings through overnight gaps; mark holdings at today's close.
   Rebalance only when exposure differs by at least the configured band, except
   a cash exit is always honored. Buy-and-hold enters once at the same first test
   open. A constant 30% benchmark helps distinguish regime selection from simply
   reducing market exposure.

Both buys and sells pay commission plus assumed slippage on actual traded
notional. Entry cost is included. Rebalancing accounts for portfolio drift and
post-cost wealth rather than subtracting a fixed fee on label changes.

The drawdown stop observes opens and closes. A close breach leads to exit at the
following open; an open breach leads to exit at that observed open. Once triggered
it stays in cash for the remainder of that run. Overnight gaps can exceed the
configured threshold. Reported maximum drawdown uses daily close equity peaks
and includes the initial capital baseline.

## Defaults and tuning

| Parameter | Default | Meaning |
|---|---:|---|
| `--train-end` | 2019-12-31 | Last permitted session for initial training |
| `--refit-every` | 0 | Fixed model; positive values enable expanding-window refits |
| `--initial-capital` | 100000 | Starting paper value in input instrument currency |
| `--cost-bps` | 10 | Commission approximation per traded notional, each side |
| `--slippage-bps` | 5 | Additional execution-friction approximation |
| `--max-exposure` | 1.0 | No leverage; maximum market allocation |
| `--defensive-exposure` | 0.3 | Defensive base allocation |
| `--target-vol` | 0.15 | Annualized volatility target used to cap exposure |
| `--min-confidence` | 0.55 | Minimum state confidence for probability-weighted base |
| `--rebalance-band` | 0.02 | Allocation difference required for ordinary rebalancing |
| `--max-drawdown` | 0.25 | Permanent strategy stop threshold within the simulation |
| `--risk-free` | 0 | Annual risk-free assumption used only in Sharpe calculation |
| `--restarts` | 3 | Training seed attempts |
| `--n-iter` | 500 | Maximum EM iterations per seed |

Cash earns zero. Sharpe uses simple portfolio returns, sample standard deviation,
252 sessions per year, and a daily equivalent of the risk-free assumption. The
annualized return is session-based, not a calendar-day CAGR. Constant-return
series have no defined Sharpe and are serialized as null.

Do not tune these settings using the final test period and then call that same
period an independent test. Use a separate validation period and fresh holdout
before making performance claims. Defaults are illustrative, not optimized.

## Output files

| File | Contents |
|---|---|
| `prices_clean.csv` | Validated OHLC |
| `features_and_signals.csv` | Features, probabilities, geometry, model cutoff, sample labels and exposure targets |
| `training_regime_statistics.csv` | Initial training-period state summaries |
| `latest_transition_matrix.csv` | Latest fitted transition probabilities, ordered low/high volatility |
| `regime_model.joblib` | Portable dictionary with latest HMM, scaler, state order and metadata |
| `*_ledger.csv` | Daily equity, cash, units, returns, turnover, fees, drawdown and stop state |
| `*_trades.csv` | Simulated fills with earlier signal dates, sizes and charged friction |
| `metrics.json` | Strategy, buy-and-hold and constant-30% results |
| `latest_paper_decision.json` | Latest forecast, paper holdings and illustrative pending next-open order |
| `run_metadata.json` | Config, training fits, versions, price hash, data checks and assumptions |
| `performance.png` | Historical states and out-of-sample equity/drawdown plots |

The latest decision's notional is estimated at the last close; it is not a fill.
Sizing must be recomputed at the eventual opening price. Loading a joblib file
can execute code: load only artifacts you generated or trust.

## Validation and remaining implementation scope

The supplied tests verify forward-filter prefix invariance, future-price
perturbation invariance, train-only scaling, transition probability sums,
exposure caps, exact fee/account accounting, next-open timing, entry costs,
initial drawdowns, drawdown-stop behavior, chronological refits, column handling,
invalid input rejection, correct simple returns, and full output generation.

The complete synthetic pipeline and all 22 tests passed during delivery. The
Yahoo Finance NIFTY download was rate-limited in this environment, so no actual
NIFTY performance is claimed. The included demo reports are synthetic. A full
expanding-window synthetic run was also exercised; see VALIDATION.md.

This implements the research and paper-simulation framework. Production order
execution needs a separate broker adapter, authentication, exchange-specific
calendar, market-status checks, discrete quantities, order rejection/partial-fill
handling, persistent cash/position reconciliation, and operational monitoring.
The current simulator assumes fractional index-proxy units, observed-open fills,
fixed friction, no cash interest and no taxes or liquidity/market-impact model.
There is no evidence here that the strategy will outperform on real markets.

Reference documentation used:
- https://hmmlearn.readthedocs.io/en/stable/api.html
- https://scikit-learn.org/stable/common_pitfalls.html
- https://ranaroussi.github.io/yfinance/advanced/multi_level_columns.html

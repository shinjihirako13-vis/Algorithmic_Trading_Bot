# Indian market monitoring without a broker account

You can run a public-data research monitor without opening a trading account.
The included mode polls Yahoo Finance's recent intraday candles through yfinance.
It requires no broker credentials and submits no orders.

**This is not guaranteed real-time exchange data.** Public delivery can be delayed,
unavailable or rate-limited. The program reports observed candle timestamps and
age, never certifies the feed as real-time, and pauses signals when observations
are stale or unavailable. Actual feed access was rate-limited during verification.

## Start

Extract the complete ZIP into one folder; the monitor imports the main script.
Install the existing requirements first:

```bash
python -m pip install -r requirements.txt
```

NIFTY 50, five-minute candles, checking once per minute:

```bash
python Q_prog1.py --public-live --symbol NIFTY50 --poll 60
```

BANK NIFTY:

```bash
python Q_prog1.py --public-live --symbol BANKNIFTY --output banknifty_monitor
```

SENSEX:

```bash
python Q_prog1.py --public-live --symbol SENSEX --output sensex_monitor
```

An NSE-listed stock:

```bash
python Q_prog1.py --public-live --symbol RELIANCE.NS --output reliance_monitor
```

BSE stocks use the provider's `.BO` symbol. Verify the ticker's identity before
interpreting results. NIFTY 50, BANK NIFTY and SENSEX are research index proxies,
not executable instruments.

Use `--once` to check once, `--interval 15m` for fifteen-minute bars, or Ctrl+C to
stop continuous polling. Use a different output folder for each ticker/interval:

```bash
python Q_prog1.py --public-live --symbol NIFTY50 --interval 15m --output nifty_15m --once
python Q_prog1.py --public-live --help
```

The entry point can also be run directly:

```bash
python indian_market_monitor.py --symbol NIFTY50 --once
```

## Model behavior

- Downloads up to 59 calendar days of available intraday history.
- Converts provider timestamps to Asia/Kolkata and preserves intraday timestamps.
- Filters regular weekday-session candles on the 09:15–15:30 IST grid. This is
  an ordinary-session window, not a verified holiday/special-session calendar.
- Discards unfinished candles and adds a ten-second completion grace.
- Builds returns and a trailing 20-bar volatility feature; the model is trained
  on the same interval that it analyzes. It does not feed five-minute returns
  into the original daily-trained HMM.
- Fits normalization and HMM on the first 20 feature sessions of received history
  and then freezes the model. Subsequent observations use forward filtering.
- Uses an annualization assumption of 252 sessions times 75 five-minute bars,
  or 25 fifteen-minute bars, per regular session. Overnight gaps are contained
  in session-opening close-to-close returns; this is an approximation.
- Saves a local model and an accumulating price cache so repeated polls preserve
  the filtering context. `--reset-model` explicitly refits using that profile's
  initial training segment; this is not automatic rolling retraining.
- Reports current state probabilities, next-bar state probabilities and a
  research exposure target. It creates no simulated fills and no broker orders.

The public feed may revise earlier candles; revised rows replace corresponding
cache rows and filtered results may change. Such results are observational
research, not a verified record of broker-executable signals.

## Reading the status

`indian_monitor_output/latest_intraday_status.json` contains:

| Field | Meaning |
|---|---|
| `observed_close` | Price at the end of the latest completed received candle |
| `latest_completed_bar_start_ist` | Candle start, not necessarily the current time |
| `observed_bar_age_minutes` inside `freshness` | Minutes since that candle ended |
| `regime` | Low-Vol or High-Vol inferred state |
| `next_bar_low_vol_probability` | Forecast probability for the next model bar |
| `current_research_target_exposure` | Current research target; null if paused |
| `guaranteed_realtime` inside `freshness` | Always false for this public-data mode |
| `action` | RESEARCH_SIGNAL_ONLY or PAUSED_NO_CURRENT_SIGNAL |

The default freshness threshold is two intervals plus two minutes (12 minutes
for five-minute candles). Change it with `--max-age-minutes`; this threshold
does not establish the provider's actual latency. After normal session hours,
during stale-data conditions, or on failures, the current target is null. During
holidays the weekday time window may still be open but the price age check will
pause stale observations.

Supporting files include `intraday_prices_cache.csv`,
`intraday_features_and_signals.csv`, `intraday_model.joblib`, and
`monitor_profile.json`. Keep your generated cache/profile/model together. Load
only model files you created or trust.

## Public feed unavailable

An empty response or rate limit is recorded as FEED_OR_MODEL_ERROR with no current
signal. Continuous mode retries at the configured interval; it does not invent
prices or substitute synthetic prices. One-shot mode exits with code 1. A feed
failure does not delete historical cache data. Repeated rate limits require
waiting for the provider or switching to an authenticated data source; polling
faster does not fix them.

## If you need actual real-time exchange data

A brokerage account is one access route, not the only route. NSE provides paid
real-time data products and lists authorized vendors. A licensed data-vendor API
can be used with a provider subscription instead of a trading account. That
requires provider-specific credentials and a separate authenticated adapter.
This public-data mode does not implement that connection.

Official references:
- NSE real-time data: https://www.nseindia.com/static/market-data/real-time-data-subscription
- NSE data products: https://www.nseindia.com/static/nse-data-and-analytics/data-information-vending
- yfinance intraday interface: https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- Yahoo data-delay information: https://help.yahoo.com/kb/finance/article-exchanges-data-delays-sln2310.html
- Research-oriented yfinance documentation: https://ranaroussi.github.io/yfinance/index.html

## Verify locally

```bash
python -m unittest -v test_bot test_monitor
```

All 22 tests passed during delivery. Actual public Indian intraday access could
not be verified here because Yahoo Finance rate-limited the request.

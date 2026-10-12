# Delivery validation

All results below describe software checks, not proven trading profitability.

| Check | Result |
|---|---|
| Python 3.12, pinned direct dependencies | Installed and exercised |
| Unit/integration suite | 22 tests passed (15 framework + 7 intraday monitor) |
| Fixed-model synthetic CLI | Passed; 3,200 OHLC rows, 1,635 evaluated test sessions |
| Expanding-window synthetic CLI | Passed; refit interval 252 sessions |
| CSV CLI roundtrip | Passed; metrics matched the generated-data run within numerical tolerance |
| Training timestamp for every test signal | Strictly earlier than signal date |
| Execution timestamp for every test fill | Uses an earlier session's signal |
| Equity reconstruction | Compounded ledger returns reconcile with portfolio wealth |
| Generated plot | Visually inspected; includes explicit synthetic source label |
| Actual NIFTY Yahoo Finance daily download | Rate-limited on three attempts; no market backtest completed |
| Public NIFTY five-minute request | Rate-limited; produced paused error status and failure exit code |
| Intraday monitor logic | Seven tests passed: completion, timezone, freshness, cache revisions, saved model, feed failure, ticker validation |

Rerun the suite from the project directory:

```bash
python -m unittest -v test_bot test_monitor
```

The tests cover inference causality, feature causality, normalization boundaries,
transition probabilities, exposure limits, cash/holdings/fees, execution timing,
initial entry costs, initial drawdown, stop behavior, expanding-window boundaries,
data validation, return conversion and output serialization.

An HMM training attempt may log a tiny likelihood decrease from floating-point
precision. The fit selector checks final likelihood stability with a 0.001
tolerance, rejects materially unstable/collapsed fits, and records each seed's
diagnostics in run_metadata.json. Reaching the iteration limit alone does not
qualify a model as stable.

The shipped demo_output and walkforward_demo folders contain synthetic-only
examples. Their figures and metadata identify that source. No real-market result,
live broker fill, performance guarantee or paper/live broker reconciliation was
verified.

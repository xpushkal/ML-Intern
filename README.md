# MLOps Batch Job — Rolling Mean Signal Pipeline

A minimal, production-grade MLOps batch job that computes a binary trading signal
from OHLCV data using a configurable rolling mean. Designed for determinism,
observability, and Docker-first deployment.

---

## Architecture

```
data.csv + config.yaml
        │
        ▼
   ┌─────────┐   validate   ┌──────────────┐   rolling    ┌──────────┐
   │  Config │ ──────────▶  │   Dataset    │ ──  mean  ─▶ │  Signal  │
   │  Loader │              │   Validator  │              │  Engine  │
   └─────────┘              └──────────────┘              └────┬─────┘
                                                               │
                                                    ┌──────────▼──────────┐
                                                    │  metrics.json        │
                                                    │  run.log             │
                                                    │  stdout (JSON)       │
                                                    └─────────────────────┘
```

---

## Requirements

- Python 3.9+
- Docker (for containerized run)

---

## Local Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

> All paths are CLI arguments — no hardcoded paths anywhere.

### 3. Inspect outputs

```bash
cat metrics.json   # structured metrics
cat run.log        # detailed execution log
```

---

## Docker Run

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

- Exit code `0` → success
- Exit code `1` → failure (error details in stdout JSON + run.log)

---

## Config (`config.yaml`)

| Key       | Type    | Description                        |
|-----------|---------|------------------------------------|
| `seed`    | int ≥ 0 | NumPy random seed for determinism  |
| `window`  | int ≥ 1 | Rolling mean window size           |
| `version` | string  | Pipeline version tag               |

```yaml
seed: 42
window: 5
version: "v1"
```

---

## Signal Logic

| Condition              | Signal |
|------------------------|--------|
| `close > rolling_mean` | `1`    |
| `close ≤ rolling_mean` | `0`    |

**Warm-up rows** (first `window - 1` rows): rolling mean is `NaN`, signal defaults to `0`.

---

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 68.49,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Available columns: ['open', 'high']"
}
```

---

## Reproducibility

Running the pipeline twice with the same `config.yaml` always produces identical
`metrics.json` output. The seed gates NumPy's global RNG; the rolling mean
computation is fully deterministic (O(n) cumulative-sum approach).

---

## Error Handling

The pipeline validates:
- Config file exists and has all required keys (`seed`, `window`, `version`)
- Input CSV exists, is non-empty, is parseable, and contains `close`
- `close` values are all numeric

On any failure: `metrics.json` is written with `status: error`, exit code is `1`.

---

## File Structure

```
.
├── run.py           # Main pipeline
├── config.yaml      # Pipeline configuration
├── data.csv         # OHLCV input data (10,000 rows)
├── requirements.txt # Pinned Python dependencies
├── Dockerfile       # Multi-stage Docker build
├── README.md        # This file
├── metrics.json     # Sample output (successful run)
└── run.log          # Sample log (successful run)
```
# ML-Intern

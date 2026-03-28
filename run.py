"""
MLOps Batch Job: Rolling Mean Signal Pipeline
Computes a binary trading signal (close > rolling_mean) on OHLCV data.
"""

import argparse
import json
import logging
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    """Configure root logger to write to both file and stdout."""
    logger = logging.getLogger("mlops")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # File handler — full detail
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Console handler — info+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str) -> dict:
    """Parse and validate config.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping (key: value pairs).")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    # Type validation
    if not isinstance(cfg["seed"], int) or cfg["seed"] < 0:
        raise ValueError(f"'seed' must be a non-negative integer, got: {cfg['seed']!r}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {cfg['window']!r}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"'version' must be a non-empty string, got: {cfg['version']!r}")

    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(input_path: str) -> pd.DataFrame:
    """
    Load and validate the OHLCV CSV.

    Handles the quirk where the file wraps each row in outer double-quotes,
    making each physical line appear as a single CSV column. We detect this
    and re-parse accordingly.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("Input file is empty.")

    # Try standard parse first; fall back to unwrapping outer quotes
    try:
        df = pd.read_csv(StringIO(raw))
        if "close" not in df.columns:
            # Possibly the outer-quoted format — unwrap and retry
            lines = [line.strip().strip('"') for line in raw.splitlines()]
            df = pd.read_csv(StringIO("\n".join(lines)))
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV parse error: {exc}") from exc

    if df.empty:
        raise ValueError("Dataset contains no rows after parsing.")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce close to numeric; non-parseable values become NaN
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    n_invalid = df["close"].isna().sum()
    if n_invalid > 0:
        raise ValueError(
            f"'close' column contains {n_invalid} non-numeric value(s). "
            "Cannot proceed."
        )

    return df


# ---------------------------------------------------------------------------
# Signal computation (pure NumPy — fast & deterministic)
# ---------------------------------------------------------------------------

def compute_signal(close: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling mean and binary signal.

    Strategy for first (window-1) rows:
        Rolling mean is NaN → those rows are excluded from signal computation.
        Signal defaults to 0 for warm-up rows (conservative / no position).

    Returns
    -------
    rolling_mean : float64 array, length == len(close)
    signal       : int8 array,    length == len(close)
    """
    n = len(close)
    rolling_mean = np.full(n, np.nan, dtype=np.float64)
    signal = np.zeros(n, dtype=np.int8)

    # Cumulative sum trick for O(n) rolling mean
    cumsum = np.cumsum(close)
    for i in range(window - 1, n):
        if i == window - 1:
            rolling_mean[i] = cumsum[i] / window
        else:
            rolling_mean[i] = (cumsum[i] - cumsum[i - window]) / window

    # Signal only on valid (non-warm-up) rows
    valid = ~np.isnan(rolling_mean)
    signal[valid] = (close[valid] > rolling_mean[valid]).astype(np.int8)

    return rolling_mean, signal


# ---------------------------------------------------------------------------
# Metrics output
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict) -> None:
    Path(output_path).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def build_success_metrics(version: str, rows: int, signal: np.ndarray,
                          latency_ms: float, seed: int) -> dict:
    # signal_rate computed only over valid (non-warm-up) rows where signal != sentinel
    # But since warm-up rows are signal=0, we include ALL rows for signal_rate
    # as per spec: mean(signal) over all rows_processed.
    return {
        "version": version,
        "rows_processed": rows,
        "metric": "signal_rate",
        "value": round(float(signal.mean()), 4),
        "latency_ms": round(latency_ms, 2),
        "seed": seed,
        "status": "success",
    }


def build_error_metrics(version: str, message: str) -> dict:
    return {
        "version": version,
        "status": "error",
        "error_message": message,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLOps batch job: rolling-mean signal pipeline"
    )
    parser.add_argument("--input",    required=True, help="Path to OHLCV CSV")
    parser.add_argument("--config",   required=True, help="Path to config.yaml")
    parser.add_argument("--output",   required=True, help="Path for metrics JSON output")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for structured log output")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    """
    Execute the full pipeline.
    Returns 0 on success, 1 on failure.
    """
    logger = setup_logging(args.log_file)
    t_start = time.perf_counter()
    version = "unknown"

    logger.info("=" * 60)
    logger.info("Job started")
    logger.info("Input   : %s", args.input)
    logger.info("Config  : %s", args.config)
    logger.info("Output  : %s", args.output)
    logger.info("Log file: %s", args.log_file)
    logger.info("=" * 60)

    try:
        # ── 1. Load & validate config ──────────────────────────────────
        logger.info("Loading config from: %s", args.config)
        cfg = load_config(args.config)
        version = cfg["version"]
        seed    = cfg["seed"]
        window  = cfg["window"]
        logger.info(
            "Config validated — version=%s | seed=%d | window=%d",
            version, seed, window
        )

        # ── 2. Set random seed ─────────────────────────────────────────
        np.random.seed(seed)
        logger.debug("NumPy random seed set to %d", seed)

        # ── 3. Load & validate dataset ─────────────────────────────────
        logger.info("Loading dataset from: %s", args.input)
        df = load_dataset(args.input)
        n_rows = len(df)
        logger.info("Dataset loaded: %d rows, %d columns", n_rows, len(df.columns))
        logger.debug("Columns: %s", list(df.columns))
        logger.debug(
            "close  min=%.2f  max=%.2f  mean=%.2f",
            df["close"].min(), df["close"].max(), df["close"].mean()
        )

        # ── 4. Rolling mean ────────────────────────────────────────────
        logger.info("Computing rolling mean (window=%d) ...", window)
        close_arr = df["close"].to_numpy(dtype=np.float64)
        rolling_mean, signal = compute_signal(close_arr, window)

        n_valid   = int((~np.isnan(rolling_mean)).sum())
        n_warmup  = n_rows - n_valid
        logger.info(
            "Rolling mean computed — warm-up rows (NaN): %d, valid rows: %d",
            n_warmup, n_valid
        )

        # ── 5. Signal generation ───────────────────────────────────────
        logger.info("Generating binary signals ...")
        n_signal_1 = int(signal.sum())
        n_signal_0 = n_rows - n_signal_1
        logger.info(
            "Signal distribution — signal=1: %d (%.2f%%)  signal=0: %d (%.2f%%)",
            n_signal_1, 100 * n_signal_1 / n_rows,
            n_signal_0, 100 * n_signal_0 / n_rows,
        )

        # ── 6. Metrics ─────────────────────────────────────────────────
        latency_ms = (time.perf_counter() - t_start) * 1000
        metrics = build_success_metrics(version, n_rows, signal, latency_ms, seed)

        write_metrics(args.output, metrics)
        logger.info("Metrics written to: %s", args.output)
        logger.info(
            "Summary — rows_processed=%d | signal_rate=%.4f | latency_ms=%.2f",
            metrics["rows_processed"],
            metrics["value"],
            metrics["latency_ms"],
        )

        logger.info("Job completed successfully")
        logger.info("=" * 60)

        # Print final metrics to stdout (Docker requirement)
        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        error_payload = build_error_metrics(version, str(exc))
        try:
            write_metrics(args.output, error_payload)
            logger.error("Error metrics written to: %s", args.output)
        except Exception as write_exc:
            logger.error("Could not write error metrics: %s", write_exc)

        logger.info("Job ended with status: error")
        logger.info("=" * 60)
        print(json.dumps(error_payload, indent=2))
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run(parse_args()))

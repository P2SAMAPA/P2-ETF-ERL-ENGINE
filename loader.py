# loader.py — Loads ETF and benchmark prices from HuggingFace
# Source repo: P2SAMAPA/p2-etf-deepwave-dl
#
# Actual files in repo:
#   data/etf_price.parquet    — ETF adjusted prices
#   data/bench_price.parquet  — AGG benchmark prices
#   data/etf_ret.parquet      — ETF daily returns (pre-computed)
#   data/bench_ret.parquet    — benchmark daily returns
#   data/macro.parquet        — macro features (used by HMM directly)

import os
import sys
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def _download(filename):
    return hf_hub_download(
        repo_id        = cfg.HF_SOURCE_REPO,
        filename       = filename,
        repo_type      = "dataset",
        token          = cfg.HF_TOKEN if cfg.HF_TOKEN else None,
        force_download = True,
    )


def load_etf_prices() -> pd.DataFrame:
    path = _download("data/etf_price.parquet")
    df   = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df   = df.sort_index()

    # Keep only our assets, in order
    available = [a for a in cfg.ASSETS if a in df.columns]
    missing   = [a for a in cfg.ASSETS if a not in df.columns]
    if missing:
        raise ValueError(f"[loader] Missing assets in source data: {missing}")

    df = df[available]
    df = df[df.index >= cfg.DATA_START].dropna(how='all')
    print(f"[loader] ETF prices: {len(df)} days "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def load_benchmark_prices() -> pd.Series:
    path = _download("data/bench_price.parquet")
    df   = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df   = df.sort_index()

    # Handle single-column or named column
    if cfg.BENCHMARK in df.columns:
        series = df[cfg.BENCHMARK]
    else:
        series = df.iloc[:, 0]
        series.name = cfg.BENCHMARK

    series = series[series.index >= cfg.DATA_START].dropna()
    print(f"[loader] Benchmark ({cfg.BENCHMARK}): {len(series)} days "
          f"({series.index[0].date()} → {series.index[-1].date()})")
    return series


def load_etf_returns() -> pd.DataFrame:
    """Load pre-computed ETF daily returns."""
    path = _download("data/etf_ret.parquet")
    df   = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df   = df.sort_index()
    available = [a for a in cfg.ASSETS if a in df.columns]
    df   = df[available]
    df   = df[df.index >= cfg.DATA_START].dropna(how='all')
    print(f"[loader] ETF returns: {len(df)} days")
    return df


def load_benchmark_returns() -> pd.Series:
    """Load pre-computed benchmark daily returns."""
    path = _download("data/bench_ret.parquet")
    df   = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df   = df.sort_index()

    if cfg.BENCHMARK in df.columns:
        series = df[cfg.BENCHMARK]
    else:
        series = df.iloc[:, 0]
        series.name = cfg.BENCHMARK

    series = series[series.index >= cfg.DATA_START].dropna()
    print(f"[loader] Benchmark returns: {len(series)} days")
    return series


def load_macro() -> pd.DataFrame:
    """
    Load pre-computed macro features from source repo.
    Used directly by HMM — no recomputation needed.
    Columns expected: yield_curve_slope, credit_spread, vol_regime,
                      real_rate_direction, risk_appetite (or similar)
    """
    path = _download("data/macro.parquet")
    df   = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df   = df.sort_index()
    df   = df[df.index >= cfg.DATA_START].dropna(how='all')
    print(f"[loader] Macro features: {len(df)} days, "
          f"cols={list(df.columns)}")
    return df


def align_all(etf_prices, benchmark, etf_returns, bench_returns):
    """Align all frames to common trading days."""
    common = (etf_prices.index
              .intersection(benchmark.index)
              .intersection(etf_returns.index)
              .intersection(bench_returns.index))
    common = common.sort_values()

    return (
        etf_prices.reindex(common),
        benchmark.reindex(common),
        etf_returns.reindex(common),
        bench_returns.reindex(common),
    )


def split_train_live(df):
    train = df[df.index <= cfg.TRAIN_END]
    live  = df[df.index >= cfg.LIVE_START]
    return train, live


def compute_daily_returns(prices):
    """Fallback — compute returns from prices if pre-computed not available."""
    return prices.pct_change().dropna()


def load_all(force_download: bool = True) -> dict:
    """
    Load and align everything in one call.

    Returns dict with keys:
        prices, benchmark, returns, bench_returns,
        train_prices, live_prices, train_bench, live_bench,
        macro (if available)
    """
    etf_prices    = load_etf_prices()
    benchmark     = load_benchmark_prices()

    # Try pre-computed returns first, fall back to computing from prices
    try:
        etf_returns   = load_etf_returns()
        bench_returns = load_benchmark_returns()
    except Exception as e:
        print(f"[loader] Pre-computed returns not found ({e}) "
              f"— computing from prices")
        etf_returns   = compute_daily_returns(etf_prices)
        bench_returns = compute_daily_returns(benchmark)

    # Align
    etf_prices, benchmark, etf_returns, bench_returns = align_all(
        etf_prices, benchmark, etf_returns, bench_returns
    )
    print(f"[loader] Aligned: {len(etf_prices)} common trading days")

    # Train/live splits
    train_prices, live_prices = split_train_live(etf_prices)
    train_bench,  live_bench  = split_train_live(benchmark)

    result = {
        'prices':        etf_prices,
        'benchmark':     benchmark,
        'returns':       etf_returns,
        'bench_returns': bench_returns,
        'train_prices':  train_prices,
        'live_prices':   live_prices,
        'train_bench':   train_bench,
        'live_bench':    live_bench,
    }

    # Load macro if available
    try:
        result['macro'] = load_macro()
    except Exception as e:
        print(f"[loader] Macro not loaded ({e})")

    return result


if __name__ == "__main__":
    data = load_all()
    print("\n── Summary ───────────────────────────────────")
    print(f"Prices:  {data['prices'].shape}")
    print(f"Returns: {data['returns'].shape}")
    print(f"Train:   {len(data['train_prices'])} days")
    print(f"Live:    {len(data['live_prices'])} days")
    print(data['prices'].tail(3))

# loader.py — Loads ETF and benchmark prices from HuggingFace
# Reads from P2SAMAPA/p2-etf-deepwave-dl (existing live price source)
# Returns clean, aligned DataFrames ready for feature computation

import os
import sys
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def load_etf_prices(force_download: bool = True) -> pd.DataFrame:
    """
    Load total-return adjusted ETF prices from HF source repo.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (trading days)
        Columns: ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'SLV']
        Values: total return adjusted closing prices
    """
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_SOURCE_REPO,
            filename="etf_prices.csv",
            repo_type="dataset",
            token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
            force_download=force_download,
        )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Keep only our assets
        available = [a for a in cfg.ASSETS if a in df.columns]
        missing   = [a for a in cfg.ASSETS if a not in df.columns]
        if missing:
            raise ValueError(f"[loader] Missing assets in source data: {missing}")

        df = df[available]

        # Filter to data range
        df = df[df.index >= cfg.DATA_START]
        df = df.dropna(how='all')

        print(f"[loader] ETF prices loaded: {len(df)} days "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df

    except Exception as e:
        raise RuntimeError(f"[loader] Failed to load ETF prices: {e}")


def load_benchmark_prices(force_download: bool = True) -> pd.Series:
    """
    Load AGG benchmark total-return adjusted prices from HF source repo.

    Returns
    -------
    pd.Series
        Index: DatetimeIndex
        Name: 'AGG'
        Values: total return adjusted closing prices
    """
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_SOURCE_REPO,
            filename="benchmark_prices.csv",
            repo_type="dataset",
            token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
            force_download=force_download,
        )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if cfg.BENCHMARK not in df.columns:
            # Try loading it as a single-column file
            series = df.iloc[:, 0]
            series.name = cfg.BENCHMARK
        else:
            series = df[cfg.BENCHMARK]

        series = series[series.index >= cfg.DATA_START]
        series = series.dropna()

        print(f"[loader] Benchmark ({cfg.BENCHMARK}) loaded: {len(series)} days "
              f"({series.index[0].date()} → {series.index[-1].date()})")
        return series

    except Exception as e:
        raise RuntimeError(f"[loader] Failed to load benchmark prices: {e}")


def align_prices(
    etf_prices: pd.DataFrame,
    benchmark: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align ETF prices and benchmark to the same trading days.
    Drops any day where either has NaN.

    Returns
    -------
    (etf_prices_aligned, benchmark_aligned)
    """
    combined = etf_prices.join(benchmark, how='inner')
    combined = combined.dropna()

    etf_aligned   = combined[cfg.ASSETS]
    bench_aligned = combined[cfg.BENCHMARK]

    print(f"[loader] Aligned: {len(combined)} common trading days")
    return etf_aligned, bench_aligned


def split_train_live(
    df: pd.DataFrame | pd.Series,
) -> tuple:
    """
    Split data into training period (up to TRAIN_END)
    and live period (from LIVE_START onwards).

    Works with both DataFrame and Series.
    """
    train = df[df.index <= cfg.TRAIN_END]
    live  = df[df.index >= cfg.LIVE_START]

    if isinstance(df, pd.DataFrame):
        print(f"[loader] Train: {len(train)} days "
              f"({train.index[0].date()} → {train.index[-1].date()})")
        print(f"[loader] Live:  {len(live)} days "
              f"({live.index[0].date()} → {live.index[-1].date()})")
    return train, live


def get_regime_slice(
    df: pd.DataFrame | pd.Series,
    regime_labels: pd.Series,
    regimes: list[int],
) -> pd.DataFrame | pd.Series:
    """
    Filter data to rows where HMM regime label is in the given list.
    Used to create regime-specific training sets for Policy A/B/C.

    Parameters
    ----------
    df            : price or feature DataFrame/Series
    regime_labels : pd.Series with same index, values = int regime labels
    regimes       : list of regime ints to keep (e.g. [4, 5, 6] for crisis)
    """
    mask = regime_labels.isin(regimes)
    aligned_mask = mask.reindex(df.index).fillna(False)
    filtered = df[aligned_mask]

    print(f"[loader] Regime slice {regimes}: "
          f"{len(filtered)} days ({len(filtered)/len(df)*100:.1f}% of data)")
    return filtered


def compute_daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compute daily percentage returns from price series.
    First row is dropped (NaN from pct_change).
    """
    returns = prices.pct_change().dropna()
    return returns


def load_all(force_download: bool = True) -> dict:
    """
    Convenience function — loads and aligns everything in one call.

    Returns
    -------
    dict with keys:
        'prices'         : pd.DataFrame — all ETF prices, aligned
        'benchmark'      : pd.Series   — AGG prices, aligned
        'returns'        : pd.DataFrame — ETF daily returns
        'bench_returns'  : pd.Series   — AGG daily returns
        'train_prices'   : pd.DataFrame
        'live_prices'    : pd.DataFrame
        'train_bench'    : pd.Series
        'live_bench'     : pd.Series
    """
    etf_prices = load_etf_prices(force_download=force_download)
    benchmark  = load_benchmark_prices(force_download=force_download)

    etf_prices, benchmark = align_prices(etf_prices, benchmark)

    train_prices, live_prices = split_train_live(etf_prices)
    train_bench,  live_bench  = split_train_live(benchmark)

    returns       = compute_daily_returns(etf_prices)
    bench_returns = compute_daily_returns(benchmark)

    return {
        'prices':        etf_prices,
        'benchmark':     benchmark,
        'returns':       returns,
        'bench_returns': bench_returns,
        'train_prices':  train_prices,
        'live_prices':   live_prices,
        'train_bench':   train_bench,
        'live_bench':    live_bench,
    }


if __name__ == "__main__":
    """Quick smoke test — run directly to verify data loads correctly."""
    data = load_all()

    print("\n── Data Summary ──────────────────────────────")
    print(f"Full price range:  {data['prices'].index[0].date()} "
          f"→ {data['prices'].index[-1].date()}")
    print(f"Total days:        {len(data['prices'])}")
    print(f"Train days:        {len(data['train_prices'])}")
    print(f"Live days:         {len(data['live_prices'])}")
    print(f"\nETF price tail:")
    print(data['prices'].tail(3).to_string())
    print(f"\nBenchmark tail:")
    print(data['benchmark'].tail(3).to_string())
    print(f"\nReturn stats (full period):")
    print(data['returns'].describe().loc[['mean','std','min','max']].to_string())

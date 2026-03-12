# score.py — Daily Signal Scorer for P2-ETF-ERL-ENGINE
# Runs on GitHub Actions immediately after predict.py.
# Looks up yesterday's signal, fetches actual prices,
# computes realised return vs AGG, and writes results back to HF.
#
# Flow:
#   1. Load signal history from HF
#   2. Find yesterday's unscored signal
#   3. Fetch actual ETF + AGG returns for that date
#   4. Compute portfolio return (dot product of weights x returns)
#   5. Mark signal as scored, update history
#   6. Update performance_summary.json
#   7. Push both files back to HF

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from loader import load_all


# ── HF Helpers ─────────────────────────────────────────────────────────────────

def _hf_download(filename, repo_id):
    return hf_hub_download(
        repo_id=repo_id, filename=filename,
        repo_type="dataset", token=cfg.HF_TOKEN, force_download=True,
    )

def _push(local_path, repo_id, repo_path):
    HfApi(token=cfg.HF_TOKEN).upload_file(
        path_or_fileobj=local_path, path_in_repo=repo_path,
        repo_id=repo_id, repo_type="dataset",
    )
    print(f"[score] Pushed {repo_path} -> {repo_id}")


# ── Return Computation ─────────────────────────────────────────────────────────

def get_actual_returns(prices, benchmark, target_date_str):
    """
    Get actual ETF + benchmark returns for a specific date.

    Parameters
    ----------
    prices         : pd.DataFrame — ETF prices (ASSETS columns)
    benchmark      : pd.Series   — AGG prices
    target_date_str: str         — 'YYYY-MM-DD'

    Returns
    -------
    dict: {asset: return} for all ETFs + AGG, or None if date not found
    """
    target = pd.Timestamp(target_date_str)
    idx    = prices.index

    if target not in idx:
        # Try to find nearest trading day
        past = idx[idx <= target]
        if len(past) < 2:
            print(f"[score] No data available for {target_date_str}")
            return None
        target   = past[-1]
        prev_day = past[-2]
    else:
        pos      = idx.get_loc(target)
        if pos == 0:
            print(f"[score] No prior day available for {target_date_str}")
            return None
        prev_day = idx[pos - 1]

    returns = {}
    for asset in cfg.ASSETS:
        if asset in prices.columns:
            p_today = prices.loc[target, asset]
            p_prev  = prices.loc[prev_day, asset]
            returns[asset] = float((p_today - p_prev) / (p_prev + 1e-8))

    # CASH always earns 0
    returns[cfg.CASH] = 0.0

    # Benchmark
    if target in benchmark.index and prev_day in benchmark.index:
        b_today = benchmark.loc[target]
        b_prev  = benchmark.loc[prev_day]
        returns[cfg.BENCHMARK] = float((b_today - b_prev) / (b_prev + 1e-8))
    else:
        returns[cfg.BENCHMARK] = 0.0

    return returns, str(target.date())


def compute_portfolio_return(allocation, actual_returns):
    """
    Compute portfolio return as weighted sum of asset returns.

    Parameters
    ----------
    allocation     : dict {asset: weight}
    actual_returns : dict {asset: return}

    Returns
    -------
    float: portfolio return for the day
    """
    port_ret = 0.0
    for asset, weight in allocation.items():
        ret      = actual_returns.get(asset, 0.0)
        port_ret += weight * ret
    return float(port_ret)


# ── Performance Summary ────────────────────────────────────────────────────────

def compute_performance_summary(history):
    """
    Compute rolling performance metrics from scored signal history.

    Returns
    -------
    dict: performance summary with Sharpe, excess return, win rate etc.
    """
    scored = [s for s in history if s.get('scored', False)]
    if len(scored) < 2:
        return {
            'n_scored':       len(scored),
            'insufficient_data': True,
        }

    port_rets  = np.array([s['portfolio_return']  for s in scored])
    bench_rets = np.array([s['benchmark_return']  for s in scored])
    excess     = port_rets - bench_rets

    # Annualised metrics (252 trading days)
    ann_port   = float((1 + port_rets).prod() ** (252 / len(port_rets)) - 1)
    ann_bench  = float((1 + bench_rets).prod() ** (252 / len(bench_rets)) - 1)
    ann_excess = ann_port - ann_bench

    # Require at least 5 days of data for std-based metrics to be meaningful
    if len(port_rets) < 5:
        port_sharpe   = float('nan')
        excess_sharpe = float('nan')
    else:
        port_sharpe = float(
            (port_rets.mean() / (port_rets.std() + 1e-8)) * np.sqrt(252)
        )
        excess_sharpe = float(
            (excess.mean() / (excess.std() + 1e-8)) * np.sqrt(252)
        )

    # Drawdown
    cumulative = (1 + port_rets).cumprod()
    peak       = np.maximum.accumulate(cumulative)
    drawdown   = (cumulative - peak) / (peak + 1e-8)
    max_dd     = float(drawdown.min())

    # Win rate vs benchmark
    win_rate = float((excess > 0).mean())

    # Rolling 20-day metrics
    recent = scored[-20:] if len(scored) >= 20 else scored
    r20    = np.array([s['portfolio_return'] for s in recent])
    e20    = np.array([s['excess_return']    for s in recent])

    # Regime breakdown
    regime_perf = {}
    for s in scored:
        rname = s.get('regime_name', 'Unknown')
        if rname not in regime_perf:
            regime_perf[rname] = []
        regime_perf[rname].append(s.get('excess_return', 0.0))

    regime_summary = {
        name: {
            'n_days':       len(rets),
            'mean_excess':  float(np.mean(rets)),
            'win_rate':     float((np.array(rets) > 0).mean()),
        }
        for name, rets in regime_perf.items()
    }

    return {
        'n_scored':          len(scored),
        'first_date':        scored[0]['date'],
        'last_date':         scored[-1]['date'],
        'computed_at':       datetime.utcnow().isoformat(),

        # Full period
        'ann_portfolio_return':  ann_port,
        'ann_benchmark_return':  ann_bench,
        'ann_excess_return':     ann_excess,
        'portfolio_sharpe':      port_sharpe,
        'excess_sharpe':         excess_sharpe,
        'max_drawdown':          max_dd,
        'win_rate_vs_bench':     win_rate,
        'total_excess_return':   float(excess.sum()),

        # Rolling 20-day
        'rolling_20d_return':    float(r20.mean() * 252) if len(r20) > 0 else 0.0,
        'rolling_20d_excess':    float(e20.mean() * 252) if len(e20) > 0 else 0.0,
        'rolling_20d_sharpe':    float(
            (r20.mean() / (r20.std() + 1e-8)) * np.sqrt(252)
        ) if len(r20) > 1 else 0.0,

        # Per-regime
        'regime_performance':    regime_summary,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg.validate()
    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)

    today_str     = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()
    print(f"\n[score] Scoring signal for {yesterday_str} (running on {today_str})")

    # ── 1. Load prices ────────────────────────────────────────────────────────
    print("\n[score] Loading prices...")
    data = load_all()

    # ── 2. Load signal history ────────────────────────────────────────────────
    print("\n[score] Loading signal history...")
    try:
        hist_path = _hf_download(cfg.SIGNAL_HISTORY_PATH, cfg.HF_RESULTS_REPO)
        with open(hist_path) as f:
            history = json.load(f)
        print(f"[score] Loaded {len(history)} signals")
    except Exception as e:
        print(f"[score] No signal history found ({e}) — nothing to score")
        return

    # ── 3. Find unscored signals ──────────────────────────────────────────────
    unscored = [s for s in history if not s.get('scored', False)]
    print(f"[score] Unscored signals: {len(unscored)}")

    if not unscored:
        print("[score] All signals already scored — nothing to do")
        return

    # ── 4. Score each unscored signal ─────────────────────────────────────────
    scored_count = 0

    for signal in unscored:
        signal_date = signal.get('date')
        if not signal_date:
            continue

        # Check if we have price data for this date
        result = get_actual_returns(data['prices'], data['benchmark'], signal_date)
        if result is None:
            print(f"[score] No price data for {signal_date} — skipping")
            continue

        actual_returns, actual_date = result

        # Single pick scoring
        pick = signal.get('pick') or signal.get('top_asset')
        if not pick:
            print(f"[score] No pick in signal for {signal_date} — skipping")
            continue
        if pick == 'CASH':
            port_ret = 0.0
        elif pick in actual_returns:
            port_ret = float(actual_returns[pick])
        else:
            print(f"[score] Pick {pick} not in returns for {signal_date} — skipping")
            continue
        bench_ret = actual_returns.get(cfg.BENCHMARK, 0.0)
        excess    = port_ret - bench_ret

        # Update signal in place
        signal['portfolio_return']  = float(port_ret)
        signal['benchmark_return']  = float(bench_ret)
        signal['excess_return']     = float(excess)
        signal['scored']            = True
        signal['scored_at']         = datetime.utcnow().isoformat()
        signal['actual_date_used']  = actual_date

        scored_count += 1
        print(f"[score] {signal_date}: port={port_ret:+.3%} "
              f"bench={bench_ret:+.3%} excess={excess:+.3%}")

    print(f"\n[score] Scored {scored_count} signal(s)")

    if scored_count == 0:
        print("[score] No new scores — exiting without push")
        return

    # ── 5. Compute performance summary ────────────────────────────────────────
    print("\n[score] Computing performance summary...")
    perf = compute_performance_summary(history)

    print(f"[score] Performance ({perf.get('n_scored', 0)} days scored):")
    if not perf.get('insufficient_data'):
        print(f"  Ann. excess return: {perf['ann_excess_return']:+.2%}")
        print(f"  Excess Sharpe:      {perf['excess_sharpe']:.3f}")
        print(f"  Win rate vs bench:  {perf['win_rate_vs_bench']:.1%}")
        print(f"  Max drawdown:       {perf['max_drawdown']:.2%}")
        print(f"  Rolling 20d excess: {perf['rolling_20d_excess']:+.2%} ann.")

    # ── 6. Save locally ───────────────────────────────────────────────────────
    history_path = os.path.join(cfg.LOCAL_TMP, "signal_history.json")
    perf_path    = os.path.join(cfg.LOCAL_TMP, "performance_summary.json")

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    with open(perf_path, 'w') as f:
        json.dump(perf, f, indent=2)

    # ── 7. Push to HF ─────────────────────────────────────────────────────────
    print("\n[score] Pushing to HuggingFace...")
    _push(history_path, cfg.HF_RESULTS_REPO, cfg.SIGNAL_HISTORY_PATH)
    _push(perf_path,    cfg.HF_RESULTS_REPO, cfg.PERFORMANCE_PATH)

    print("\n[score] Done")
    return perf


if __name__ == "__main__":
    main()

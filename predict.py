# predict.py — Daily Signal Generation for P2-ETF-ERL-ENGINE
# Runs on GitHub Actions (CPU only, ~2 min).
# Loads all models from HF, assembles today's state vector,
# runs ensemble forward pass + Kelly sizing, pushes signal to HF.

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from loader import load_all
from features import compute_hmm_features, load_or_compute_hmm_features, current_sharpe
from hmm_train import RegimeDetector
from ensemble import EnsemblePolicy
from memory import Rulebook
from kelly import format_allocation


# ── Loader Helpers ─────────────────────────────────────────────────────────────

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
    print(f"[predict] Pushed {repo_path} -> {repo_id}")


# ── State Helpers ──────────────────────────────────────────────────────────────

def get_todays_embedding(feature_cache):
    today = pd.Timestamp(date.today())
    if today in feature_cache.index:
        return feature_cache.loc[today].values.astype(np.float32)
    past = feature_cache[feature_cache.index <= today]
    if len(past) == 0:
        raise ValueError("[predict] No embedding available")
    print(f"[predict] Using embedding from {past.index[-1].date()}")
    return past.iloc[-1].values.astype(np.float32)

def get_current_weights(history):
    if not history:
        return np.ones(cfg.N_ASSETS) / cfg.N_ASSETS
    alloc = history[-1].get('allocation', {})
    w = np.array([alloc.get(a, 0.0) for a in cfg.ALL_ASSETS], dtype=np.float32)
    return w / (w.sum() + 1e-8)

def compute_rolling_sharpe_from_history(history):
    if len(history) < cfg.KELLY_SHARPE_WINDOW:
        return 0.0
    rets = [s.get('portfolio_return', 0.0) for s in history[-cfg.KELLY_SHARPE_WINDOW:]
            if s.get('portfolio_return') is not None]
    if len(rets) < 5:
        return 0.0
    r = np.array(rets)
    return float((r.mean() / (r.std() + 1e-8)) * np.sqrt(252))


# ── Signal History ─────────────────────────────────────────────────────────────

def load_signal_history():
    try:
        path = _hf_download(cfg.SIGNAL_HISTORY_PATH, cfg.HF_RESULTS_REPO)
        with open(path) as f:
            h = json.load(f)
        print(f"[predict] History loaded: {len(h)} records")
        return h
    except Exception as e:
        print(f"[predict] No history ({e}) — starting fresh")
        return []

def append_and_trim(history, signal):
    history.append(signal)
    return history[-cfg.MAX_HISTORY_RECORDS:]


# ── Signal Construction ────────────────────────────────────────────────────────

def build_signal(today_str, regime_output, ensemble_output,
                 rolling_sharpe, current_weights, active_rules, basis='live'):
    allocation = ensemble_output['allocation']
    kelly_info = ensemble_output['kelly_info']
    gates      = ensemble_output['gates']
    new_w      = ensemble_output['final_weights']
    turnover   = float(np.abs(new_w - current_weights).sum())
    top_asset  = max(allocation, key=allocation.get) if allocation else 'CASH'

    return {
        'date':              today_str,
        'generated_at':      datetime.utcnow().isoformat(),
        'basis':             basis,
        'allocation':        allocation,
        'top_asset':         top_asset,
        'turnover':          turnover,
        'regime':            regime_output['regime'],
        'regime_name':       regime_output['regime_name'],
        'crisis_prob':       regime_output['crisis_prob'],
        'transition_entropy': regime_output['transition_entropy'],
        'hmm_probs': {
            str(k): float(regime_output['probs'][k])
            for k in range(cfg.HMM_N_STATES)
        },
        'gate_A':            gates['A'],
        'gate_B':            gates['B'],
        'gate_C':            gates['C'],
        'kelly_fraction':    kelly_info['fraction'],
        'kelly_regime_scalar':    kelly_info['regime_scalar'],
        'kelly_agreement_scalar': kelly_info['agreement_scalar'],
        'kelly_sharpe_scalar':    kelly_info['sharpe_scalar'],
        'rolling_sharpe':    rolling_sharpe,
        'n_active_rules':    len(active_rules),
        'active_rule_summary': [
            r.get('action', r.get('rationale', ''))[:80]
            for r in active_rules[:3]
        ],
        'portfolio_return':  None,
        'benchmark_return':  None,
        'excess_return':     None,
        'scored':            False,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg.validate()
    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)
    today_str = date.today().isoformat()
    print(f"\n[predict] Generating signal for {today_str}")

    # 1. Prices
    data = load_all()

    # 2. HMM regime
    det_path = _hf_download("models/regime_detector.pkl", cfg.HF_MODELS_REPO)
    detector = RegimeDetector.load(det_path)
    hmm_feats = load_or_compute_hmm_features(data)
    regime_output = detector.predict(hmm_feats.tail(cfg.TFT_CONTEXT_LENGTH))
    # Debug: print full regime probs
    print(f"[predict] HMM probs: { {k: f'{v:.3f}' for k,v in enumerate(regime_output['probs'])} }")
    print(f"[predict] Dominant regime id: {regime_output['regime']} = {regime_output['regime_name']}")
    print(f"[predict] POLICY_A_REGIMES (crisis): {cfg.POLICY_A_REGIMES}")
    print(f"[predict] Regime: {regime_output['regime_name']} "
          f"(id={regime_output['regime']} crisis_p={regime_output['crisis_prob']:.1%} "
          f"entropy={regime_output['transition_entropy']:.3f})")

    # 3. TFT embedding
    cache_path = _hf_download(cfg.FEATURE_CACHE_PATH, cfg.HF_MODELS_REPO)
    feature_cache = pd.read_parquet(cache_path)
    feature_cache.index = pd.to_datetime(feature_cache.index)
    tft_embedding = get_todays_embedding(feature_cache)

    # 4. History + portfolio state
    history         = load_signal_history()
    current_weights = get_current_weights(history)
    rolling_sharpe  = compute_rolling_sharpe_from_history(history)
    print(f"[predict] Rolling Sharpe: {rolling_sharpe:.3f}")

    # 5. Rulebook
    rulebook     = Rulebook.load_from_hf()
    active_rules = rulebook.get_rules_for_regime(regime_output['regime'])
    print(f"[predict] Active rules: {len(active_rules)}")

    # 6. State vector (80-dim)
    hmm_probs = regime_output['probs']
    state = np.concatenate([
        tft_embedding,
        hmm_probs,
        current_weights,
        np.array([rolling_sharpe], dtype=np.float32),
    ]).astype(np.float32)
    assert state.shape[0] == cfg.DDPG_STATE_DIM

    # 7. Ensemble forward pass
    ensemble = EnsemblePolicy()
    ensemble_output = ensemble.forward(
        state=state, hmm_probs=hmm_probs,
        transition_entropy=regime_output['transition_entropy'],
        rolling_sharpe=rolling_sharpe,
    )
    print(f"[predict] Kelly: {ensemble_output['kelly_info']['fraction']:.3f} | "
          f"Allocation: {ensemble_output['allocation']}")

    # 8. Build + store signal
    signal  = build_signal(today_str, regime_output, ensemble_output,
                           rolling_sharpe, current_weights, active_rules)
    history = append_and_trim(history, signal)

    # 9. Save + push
    signal_path  = os.path.join(cfg.LOCAL_TMP, "latest_signal.json")
    history_path = os.path.join(cfg.LOCAL_TMP, "signal_history.json")
    with open(signal_path,  'w') as f: json.dump(signal,  f, indent=2)
    with open(history_path, 'w') as f: json.dump(history, f, indent=2)

    _push(signal_path,  cfg.HF_RESULTS_REPO, cfg.LATEST_SIGNAL_PATH)
    _push(history_path, cfg.HF_RESULTS_REPO, cfg.SIGNAL_HISTORY_PATH)

    # 10. Summary
    print(f"\n{'='*50}")
    print(f"Signal: {today_str} | {signal['regime_name']}")
    print(f"Kelly:  {signal['kelly_fraction']:.3f} | "
          f"Turnover: {signal['turnover']:.1%}")
    for asset, w in list(signal['allocation'].items())[:4]:
        print(f"  {asset:<6}: {w:.1%}")
    print(f"{'='*50}")
    print("[predict] Done")
    return signal

if __name__ == "__main__":
    main()

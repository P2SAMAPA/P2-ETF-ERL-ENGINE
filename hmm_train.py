# hmm_train.py — HMM Regime Detector for P2-ETF-ERL-ENGINE
# Trains an 8-state Hidden Markov Model on macro features to label
# market regimes. Saves model + scaler + regime history to HF.
#
# Run on Kaggle (CPU is fine — HMM is fast):
#   python hmm_train.py
#
# Outputs pushed to HF_MODELS_REPO:
#   models/hmm_model.pkl
#   models/regime_scaler.pkl
#
# Outputs pushed to HF_RESULTS_REPO:
#   results/regime_history.csv

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from hmmlearn import hmm
from huggingface_hub import HfApi, hf_hub_download
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from loader import load_all
from features import compute_hmm_features, FeatureScaler


# ── HuggingFace Push ───────────────────────────────────────────────────────────

def push_to_hf(local_path: str, repo_id: str, repo_path: str):
    api = HfApi(token=cfg.HF_TOKEN)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[HF] Pushed {repo_path} → {repo_id}")


# ── HMM Training ──────────────────────────────────────────────────────────────

def train_hmm(X_scaled: np.ndarray) -> hmm.GaussianHMM:
    """
    Train a Gaussian HMM with cfg.HMM_N_STATES latent states.
    Tries multiple random seeds and keeps the best log-likelihood.

    Parameters
    ----------
    X_scaled : np.ndarray shape (T, n_features) — normalised macro features

    Returns
    -------
    Fitted GaussianHMM model
    """
    best_model  = None
    best_score  = -np.inf
    n_attempts  = 5

    print(f"[HMM] Training {cfg.HMM_N_STATES}-state HMM "
          f"on {X_scaled.shape[0]} days × {X_scaled.shape[1]} features")

    for seed in range(n_attempts):
        try:
            model = hmm.GaussianHMM(
                n_components   = cfg.HMM_N_STATES,
                covariance_type = cfg.HMM_COVARIANCE,
                n_iter          = cfg.HMM_N_ITER,
                tol             = cfg.HMM_TOL,
                random_state    = cfg.RANDOM_SEED + seed,
                verbose         = False,
            )
            model.fit(X_scaled)
            score = model.score(X_scaled)

            print(f"[HMM] Seed {seed}: log-likelihood = {score:.2f}")

            if score > best_score:
                best_score = score
                best_model = model

        except Exception as e:
            print(f"[HMM] Seed {seed} failed: {e}")
            continue

    if best_model is None:
        raise RuntimeError("[HMM] All training attempts failed")

    print(f"[HMM] Best log-likelihood: {best_score:.2f}")
    return best_model


# ── Regime Labelling ───────────────────────────────────────────────────────────

def label_regimes(
    model: hmm.GaussianHMM,
    X_scaled: np.ndarray,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Predict regime labels for every day in the dataset.

    Returns
    -------
    pd.Series
        Index = DatetimeIndex
        Values = int regime labels (0 to HMM_N_STATES-1)
        Name = 'regime'
    """
    labels = model.predict(X_scaled)
    series = pd.Series(labels, index=index, name='regime')
    return series


def get_regime_probs(
    model: hmm.GaussianHMM,
    X_scaled: np.ndarray,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Get posterior regime probabilities for every day.
    Used as DDPG state input (8-dim probability vector).

    Returns
    -------
    pd.DataFrame
        Columns = [0, 1, ..., HMM_N_STATES-1]
        Values  = P(regime=k | observations up to t)
    """
    probs = model.predict_proba(X_scaled)
    df    = pd.DataFrame(
        probs,
        index   = index,
        columns = list(range(cfg.HMM_N_STATES)),
    )
    return df


def characterise_regimes(
    model: hmm.GaussianHMM,
    scaler: FeatureScaler,
    feature_names: list,
) -> dict:
    """
    Characterise each regime by its mean feature values (in original scale).
    Used to assign human-readable names post-training.

    Returns
    -------
    dict: {regime_id: {feature: mean_value, ...}}
    """
    # Inverse transform means back to original scale
    means_scaled = model.means_
    means_orig   = scaler.scaler.inverse_transform(means_scaled)

    characteristics = {}
    for k in range(cfg.HMM_N_STATES):
        characteristics[k] = {
            feat: float(means_orig[k, i])
            for i, feat in enumerate(feature_names)
        }
    return characteristics


def auto_name_regimes(characteristics: dict) -> dict:
    """
    Attempt to auto-assign regime names based on characteristic values.
    Uses heuristics based on yield curve slope and vol regime.

    Returns
    -------
    dict: {regime_id: name_string}
    """
    names = {}
    for k, feats in characteristics.items():
        slope   = feats.get('yield_curve_slope', 0)
        credit  = feats.get('credit_spread', 0)
        vol     = feats.get('vol_regime', 0)
        real_r  = feats.get('real_rate_direction', 0)
        risk    = feats.get('risk_appetite', 0)

        # Simple heuristic rules
        if vol > 0.015:
            if credit > 0:
                name = 'Acute Crisis'
            else:
                name = 'Risk Off'
        elif credit > 0.002:
            name = 'Credit Stress'
        elif slope < -0.001:
            name = 'Curve Flattening'
        elif real_r < -0.001:
            name = 'Late Cycle'
        elif risk > 0.002:
            name = 'Mid Cycle Growth'
        elif vol < 0.005:
            name = 'Low Vol Expansion'
        else:
            name = 'Recovery'

        names[k] = name

    return names


def build_regime_history(
    labels: pd.Series,
    probs:  pd.DataFrame,
    regime_names: dict,
) -> pd.DataFrame:
    """
    Build a full regime history DataFrame for dashboard display.

    Returns
    -------
    pd.DataFrame with columns:
        date, regime, regime_name, p0..p7, transition_entropy
    """
    df = pd.DataFrame(index=labels.index)
    df['date']         = labels.index.strftime('%Y-%m-%d')
    df['regime']       = labels.values
    df['regime_name']  = labels.map(regime_names)

    # Add probability columns
    for k in range(cfg.HMM_N_STATES):
        df[f'p{k}'] = probs[k].values

    # Transition entropy — high entropy = uncertain regime
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=1)
    df['transition_entropy'] = entropy.values

    return df.reset_index(drop=True)


# ── Regime Statistics ──────────────────────────────────────────────────────────

def compute_regime_stats(
    labels: pd.Series,
    returns: pd.DataFrame,
    regime_names: dict,
) -> pd.DataFrame:
    """
    Compute per-regime asset return statistics.
    Useful for understanding which ETFs perform best in each regime.

    Returns
    -------
    pd.DataFrame
        Index = regime names
        Columns = assets + ['count', 'pct_time']
    """
    daily_ret = returns.pct_change() if returns.columns[0] in cfg.ASSETS \
                else returns

    stats_rows = []
    for k in range(cfg.HMM_N_STATES):
        mask      = labels == k
        regime_ret = daily_ret[mask]
        n_days    = mask.sum()

        row = {
            'regime':   k,
            'name':     regime_names.get(k, f'Regime {k}'),
            'count':    int(n_days),
            'pct_time': float(n_days / len(labels) * 100),
        }
        for asset in cfg.ASSETS:
            if asset in regime_ret.columns:
                row[f'{asset}_mean_ret'] = float(
                    regime_ret[asset].mean() * 252
                )  # annualised
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows).set_index('regime')
    return stats_df


# ── Inference (used daily by predict.py) ──────────────────────────────────────

class RegimeDetector:
    """
    Lightweight inference wrapper — loaded by predict.py on GitHub Actions.
    CPU-only, fast (<1 second per day).
    """

    def __init__(self, model: hmm.GaussianHMM, scaler: FeatureScaler,
                 regime_names: dict):
        self.model        = model
        self.scaler       = scaler
        self.regime_names = regime_names

    def predict(self, macro_features: pd.DataFrame) -> dict:
        """
        Predict today's regime from recent macro features.

        Parameters
        ----------
        macro_features : pd.DataFrame — last H days of macro features

        Returns
        -------
        dict with keys:
            regime          : int
            regime_name     : str
            probs           : np.ndarray (HMM_N_STATES,)
            transition_entropy : float
            crisis_prob     : float — P(crisis regime)
        """
        X = self.scaler.transform(macro_features).values
        labels = self.model.predict(X)
        probs  = self.model.predict_proba(X)

        latest_regime = int(labels[-1])
        latest_probs  = probs[-1]

        entropy = float(
            -(latest_probs * np.log(latest_probs + 1e-8)).sum()
        )

        crisis_regimes = cfg.POLICY_A_REGIMES
        crisis_prob    = float(
            sum(latest_probs[k] for k in crisis_regimes)
        )

        return {
            'regime':              latest_regime,
            'regime_name':         self.regime_names.get(
                                       latest_regime, f'Regime {latest_regime}'
                                   ),
            'probs':               latest_probs,
            'transition_entropy':  entropy,
            'crisis_prob':         crisis_prob,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) \
            if os.path.dirname(path) else None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[RegimeDetector] Saved → {path}")

    @staticmethod
    def load(path: str) -> 'RegimeDetector':
        with open(path, 'rb') as f:
            detector = pickle.load(f)
        print(f"[RegimeDetector] Loaded ← {path}")
        return detector


# ── Visualisation ──────────────────────────────────────────────────────────────

def plot_regime_history(
    labels: pd.Series,
    regime_names: dict,
    prices: pd.DataFrame,
    save_path: str,
):
    """
    Plot regime labels over time alongside AGG price.
    Saved as PNG for dashboard.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Regime timeline
    colours = plt.cm.Set3(np.linspace(0, 1, cfg.HMM_N_STATES))
    for k in range(cfg.HMM_N_STATES):
        mask = labels == k
        ax1.fill_between(
            labels.index,
            k, k + 1,
            where=mask.values,
            color=colours[k],
            alpha=0.8,
            label=regime_names.get(k, f'R{k}'),
        )
    ax1.set_yticks(np.arange(cfg.HMM_N_STATES) + 0.5)
    ax1.set_yticklabels(
        [regime_names.get(k, f'R{k}') for k in range(cfg.HMM_N_STATES)],
        fontsize=8
    )
    ax1.set_ylabel('Regime')
    ax1.set_title('HMM Market Regime History')
    ax1.legend(loc='upper left', fontsize=7, ncol=4)

    # TLT and GLD prices for context
    for asset, colour in [('TLT', 'steelblue'), ('GLD', 'goldenrod')]:
        if asset in prices.columns:
            norm = prices[asset] / prices[asset].iloc[0] * 100
            ax2.plot(prices.index, norm, label=asset,
                     color=colour, linewidth=1)
    ax2.set_ylabel('Normalised Price (base=100)')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=8)
    ax2.set_title('ETF Prices (normalised)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[HMM] Regime plot saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg.validate()

    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[HMM] Loading data...")
    data = load_all(force_download=True)

    # ── 2. Compute HMM features ───────────────────────────────────────────────
    print("\n[HMM] Computing macro features...")
    hmm_feats = compute_hmm_features(data['prices'], data['benchmark'])

    # ── 3. Scale features (fit on train, apply to all) ────────────────────────
    print("\n[HMM] Scaling features...")
    scaler      = FeatureScaler()
    train_feats = hmm_feats[hmm_feats.index <= cfg.TRAIN_END]
    scaler.fit(train_feats)

    X_all    = scaler.transform(hmm_feats).values
    X_train  = scaler.transform(train_feats).values

    # ── 4. Train HMM on training data ─────────────────────────────────────────
    print("\n[HMM] Training...")
    model = train_hmm(X_train)

    # ── 5. Label all regimes (train + live) ───────────────────────────────────
    print("\n[HMM] Labelling regimes...")
    labels = label_regimes(model, X_all, hmm_feats.index)
    probs  = get_regime_probs(model, X_all, hmm_feats.index)

    # ── 6. Characterise and name regimes ──────────────────────────────────────
    characteristics = characterise_regimes(
        model, scaler, list(hmm_feats.columns)
    )
    regime_names = auto_name_regimes(characteristics)

    print("\n── Regime Characteristics ────────────────────")
    for k, name in regime_names.items():
        count = (labels == k).sum()
        pct   = count / len(labels) * 100
        print(f"  Regime {k} ({name:20s}): {count:4d} days ({pct:.1f}%)")

    # ── 7. Regime stats ───────────────────────────────────────────────────────
    regime_stats = compute_regime_stats(
        labels, data['returns'], regime_names
    )
    print("\n── Per-Regime Annualised Returns ─────────────")
    print(regime_stats[['name', 'count', 'pct_time'] +
                        [f'{a}_mean_ret' for a in cfg.ASSETS
                         if f'{a}_mean_ret' in regime_stats.columns]
                        ].to_string())

    # ── 8. Build regime history ───────────────────────────────────────────────
    regime_history = build_regime_history(labels, probs, regime_names)

    # ── 9. Create RegimeDetector wrapper ──────────────────────────────────────
    detector = RegimeDetector(model, scaler, regime_names)

    # ── 10. Save locally ──────────────────────────────────────────────────────
    print("\n[HMM] Saving outputs...")

    hmm_path      = os.path.join(cfg.LOCAL_TMP, "hmm_model.pkl")
    scaler_path   = os.path.join(cfg.LOCAL_TMP, "regime_scaler.pkl")
    detector_path = os.path.join(cfg.LOCAL_TMP, "regime_detector.pkl")
    history_path  = os.path.join(cfg.LOCAL_TMP, "regime_history.csv")
    stats_path    = os.path.join(cfg.LOCAL_TMP, "regime_stats.json")
    plot_path     = os.path.join(cfg.LOCAL_TMP, "regime_plot.png")

    with open(hmm_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[HMM] Model saved → {hmm_path}")

    scaler.save(scaler_path)
    detector.save(detector_path)

    regime_history.to_csv(history_path, index=False)
    print(f"[HMM] Regime history saved → {history_path}")

    # Regime names + stats as JSON
    stats_out = {
        'regime_names':       {str(k): v for k, v in regime_names.items()},
        'characteristics':    {str(k): v for k, v in characteristics.items()},
        'trained_at':         datetime.utcnow().isoformat(),
        'n_training_days':    int(len(train_feats)),
        'n_total_days':       int(len(hmm_feats)),
        'live_start':         cfg.LIVE_START,
    }
    with open(stats_path, 'w') as f:
        json.dump(stats_out, f, indent=2)

    plot_regime_history(labels, regime_names, data['prices'], plot_path)

    # ── 11. Push to HuggingFace ───────────────────────────────────────────────
    print("\n[HMM] Pushing to HuggingFace...")

    # Models repo
    push_to_hf(hmm_path,      cfg.HF_MODELS_REPO, cfg.HMM_MODEL_PATH)
    push_to_hf(scaler_path,   cfg.HF_MODELS_REPO, cfg.REGIME_SCALER_PATH)
    push_to_hf(detector_path, cfg.HF_MODELS_REPO, "models/regime_detector.pkl")
    push_to_hf(stats_path,    cfg.HF_MODELS_REPO, "models/regime_stats.json")

    # Results repo
    push_to_hf(history_path,  cfg.HF_RESULTS_REPO, cfg.REGIME_HISTORY_PATH)
    push_to_hf(plot_path,     cfg.HF_RESULTS_REPO, "results/regime_plot.png")

    print("\n[HMM] ✅ Complete — regime detector ready")
    print(f"      {cfg.HMM_N_STATES} regimes identified across "
          f"{len(hmm_feats)} days")
    print(f"      Model pushed to {cfg.HF_MODELS_REPO}")
    print(f"      History pushed to {cfg.HF_RESULTS_REPO}")


if __name__ == "__main__":
    main()

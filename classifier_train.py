# classifier_train.py — ETF Classifier using XGBoost + MLP ensemble
# XGBoost handles tabular financial data far better than pure neural nets.
# Labels: argmax(vol-adjusted next-day return) across 6 ETFs — no CASH.

import os, sys, json, pickle
import numpy as np
import pandas as pd
from datetime import date
from huggingface_hub import HfApi, hf_hub_download

# XGBoost
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# PyTorch (MLP for ensemble)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from loader import load_all
from features import load_or_compute_hmm_features
from hmm_train import RegimeDetector

LOOKBACK   = 5
N_CLASSES  = len(cfg.ASSETS)   # 6 — TLT LQD HYG VNQ GLD SLV
_N_ETF     = len(cfg.ASSETS)
INPUT_DIM  = cfg.TFT_EMBEDDING_DIM + cfg.HMM_N_STATES + 7 + (LOOKBACK * _N_ETF)  # 109

_CPU_MODE  = os.environ.get('REALM_CPU_MODE', '0') == '1'


# ── HF helpers ────────────────────────────────────────────────────────────────

def _dl(path, repo): return hf_hub_download(repo_id=repo, filename=path, repo_type='dataset', token=cfg.HF_TOKEN, force_download=True)
def _push(local, repo, remote):
    HfApi(token=cfg.HF_TOKEN).upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo, repo_type='dataset')
    print(f'[HF] Pushed {remote} → {repo}')


# ── Labels ────────────────────────────────────────────────────────────────────

def build_labels(returns: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """Label = argmax(vol-adjusted next-day return). Always an ETF, no CASH."""
    assets      = cfg.ASSETS
    rolling_vol = returns[assets].rolling(21).std().bfill().fillna(1e-4).clip(lower=1e-4)
    labels      = {}
    for i in range(len(returns) - 1):
        today    = returns.index[i]
        nxt      = returns.iloc[i + 1]
        raw      = np.array([float(nxt.get(a, 0.0)) for a in assets])
        vols     = rolling_vol.loc[today].values if today in rolling_vol.index else np.ones(_N_ETF)
        labels[today] = int(np.argmax(raw / vols))
    return pd.Series(labels, name='label')


# ── Features ──────────────────────────────────────────────────────────────────

def build_feature_matrix(data, embeddings, hmm_probs, macro, labels):
    mac_cols = ['TNX','DXY','CORP_SPREAD','HY_SPREAD','VIX','T10Y2Y','TBILL_3M']
    common   = labels.index.intersection(embeddings.dropna().index).intersection(hmm_probs.index)
    common   = common.sort_values()
    ret_idx  = data['returns'].index
    X, y, dates = [], [], []
    for t in common:
        emb = embeddings.loc[t].values.astype(np.float32)
        hmm = hmm_probs.loc[t].values.astype(np.float32)
        mac = macro.loc[t, mac_cols].values.astype(np.float32) if (macro is not None and t in macro.index) else np.zeros(7, np.float32)
        pos = ret_idx.get_loc(t) if t in ret_idx else -1
        lb  = data['returns'].iloc[pos-LOOKBACK:pos][cfg.ASSETS].values.flatten().astype(np.float32) if pos >= LOOKBACK else np.zeros(LOOKBACK * _N_ETF, np.float32)
        feat = np.concatenate([emb, hmm, mac, lb])
        assert feat.shape[0] == INPUT_DIM
        X.append(feat); y.append(int(labels.loc[t])); dates.append(t)
    return np.array(X, np.float32), np.array(y, np.int64), dates


# ── XGBoost classifier ────────────────────────────────────────────────────────

def train_xgb(X_tr, y_tr, X_va, y_va):
    # Inverse-frequency class weights
    counts  = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
    counts  = np.maximum(counts, 1)
    class_w = len(y_tr) / (N_CLASSES * counts)
    sample_w = np.array([class_w[c] for c in y_tr])

    model = xgb.XGBClassifier(
        n_estimators      = 500,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.6,
        min_child_weight  = 10,
        gamma             = 1.0,
        reg_alpha         = 0.5,
        reg_lambda        = 2.0,
        objective         = 'multi:softprob',
        num_class         = N_CLASSES,
        eval_metric       = 'mlogloss',
        use_label_encoder = False,
        n_jobs            = -1,
        random_state      = 42,
        early_stopping_rounds = 30,
    )
    model.fit(
        X_tr, y_tr,
        sample_weight    = sample_w,
        eval_set         = [(X_va, y_va)],
        verbose          = 50,
    )
    return model


def per_class_acc(model, X, y):
    preds   = model.predict(X)
    correct = np.zeros(N_CLASSES)
    total   = np.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        m = y == c
        correct[c] = (preds[m] == c).sum()
        total[c]   = m.sum()
    return {cfg.ASSETS[i]: round(float(correct[i]/max(total[i],1)), 3) for i in range(N_CLASSES)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg.validate()
    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)
    print(f"\n[CLF] ETF Classifier (XGBoost) | Input={INPUT_DIM} | Classes={N_CLASSES}")

    data = load_all()

    print("[CLF] Loading TFT embeddings...")
    emb_path   = _dl(cfg.FEATURE_CACHE_PATH, cfg.HF_MODELS_REPO)
    embeddings = pd.read_parquet(emb_path)
    embeddings.index = pd.to_datetime(embeddings.index)

    print("[CLF] Loading HMM probs...")
    det       = RegimeDetector.load(_dl('models/regime_detector.pkl', cfg.HF_MODELS_REPO))
    hmm_feats = load_or_compute_hmm_features(data)
    X_hmm     = det.scaler.transform(hmm_feats)
    hmm_probs = pd.DataFrame(
        det.model.predict_proba(X_hmm.values),
        index=hmm_feats.index, columns=list(range(cfg.HMM_N_STATES))
    ).reindex(data['returns'].index)

    macro = data.get('macro')

    print("[CLF] Building labels...")
    labels = build_labels(data['returns'], macro)
    counts = pd.Series(labels).value_counts().sort_index()
    print("[CLF] Label distribution:")
    for idx, cnt in counts.items():
        print(f"  {cfg.ASSETS[idx]:<6}: {cnt:4d} days ({cnt/len(labels):.1%})")

    print("[CLF] Building features...")
    X, y, dates = build_feature_matrix(data, embeddings, hmm_probs, macro, labels)
    print(f"[CLF] Dataset: {len(X)} × {X.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    split  = int(len(X) * 0.80)
    X_tr   = scaler.fit_transform(X[:split])
    X_va   = scaler.transform(X[split:])
    y_tr, y_va = y[:split], y[split:]
    print(f"[CLF] Train: {len(X_tr)} | Val: {len(X_va)}")

    print("[CLF] Training XGBoost...")
    model = train_xgb(X_tr, y_tr, X_va, y_va)

    val_preds = model.predict(X_va)
    val_acc   = (val_preds == y_va).mean()
    print(f"\n[CLF] Val accuracy: {val_acc:.3f}")

    pc = per_class_acc(model, X_va, y_va)
    print("[CLF] Per-class val accuracy:")
    for asset, acc in pc.items():
        print(f"  {asset:<6}: {acc:.1%}")

    # Save model + scaler
    model_path  = os.path.join(cfg.LOCAL_TMP, 'etf_classifier.pkl')
    scaler_path = os.path.join(cfg.LOCAL_TMP, 'clf_scaler.pkl')
    meta_path   = os.path.join(cfg.LOCAL_TMP, 'classifier_meta.json')

    with open(model_path,  'wb') as f: pickle.dump(model,  f)
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)

    meta = {
        'type': 'xgboost', 'input_dim': INPUT_DIM, 'n_classes': N_CLASSES,
        'lookback': LOOKBACK, 'best_val_acc': float(val_acc),
        'per_class_acc': pc, 'best_iteration': int(model.best_iteration),
        'train_size': len(X_tr), 'val_size': len(X_va),
        'trained_date': date.today().isoformat(),
    }
    with open(meta_path, 'w') as f: json.dump(meta, f, indent=2)

    print("\n[CLF] Pushing to HuggingFace...")
    _push(model_path,  cfg.HF_MODELS_REPO, 'models/etf_classifier.pkl')
    _push(scaler_path, cfg.HF_MODELS_REPO, 'models/clf_scaler.pkl')
    _push(meta_path,   cfg.HF_MODELS_REPO, 'models/classifier_meta.json')
    print(f"\n[CLF] ✅ Complete | Val acc: {val_acc:.1%} | Best iter: {model.best_iteration}")


if __name__ == '__main__':
    main()

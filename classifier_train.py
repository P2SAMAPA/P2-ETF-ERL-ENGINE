# classifier_train.py — Option B: Direct ETF Classification
# Trains a supervised Transformer classifier to predict the best ETF for next day.
# Label = argmax(next-day returns) across TLT/LQD/HYG/VNQ/GLD/SLV/CASH
# CASH label = day where ALL ETFs had negative returns OR VIX > 25
#
# Architecture:
#   Input: [TFT embedding (64) + HMM probs (8) + macro (7) + 5d lookback returns (35)] = 114 dims
#   → Linear projection → Transformer encoder (4 heads, 2 layers)
#   → Classification head → 7 classes (one per asset)
#   → CrossEntropy loss + Label smoothing

import os, sys, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import date
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from loader import load_all
from features import load_or_compute_hmm_features, FeatureScaler
from hmm_train import RegimeDetector

_CPU_MODE = os.environ.get('REALM_CPU_MODE', '0') == '1'
DEVICE     = torch.device('cpu')
MAX_EPOCHS = 20 if _CPU_MODE else 60
LR         = 3e-4
BATCH_SIZE = 64
LOOKBACK   = 5          # days of return history in state
N_CLASSES  = cfg.N_ASSETS  # 7
_N_ETF     = len(cfg.ASSETS)   # 6 (excludes CASH)
INPUT_DIM  = cfg.TFT_EMBEDDING_DIM + cfg.HMM_N_STATES + 7 + (LOOKBACK * _N_ETF)  # 64+8+7+30=109
D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 2
DROPOUT    = 0.1
PATIENCE   = 8


# ── Helper ────────────────────────────────────────────────────────────────────

def _hf_download(filename, repo_id):
    return hf_hub_download(
        repo_id=repo_id, filename=filename,
        repo_type='dataset', token=cfg.HF_TOKEN, force_download=True,
    )

def _push(local_path, repo_id, repo_path):
    HfApi(token=cfg.HF_TOKEN).upload_file(
        path_or_fileobj=local_path, path_in_repo=repo_path,
        repo_id=repo_id, repo_type='dataset',
    )
    print(f'[HF] Pushed {repo_path} → {repo_id}')


# ── Label Generation ──────────────────────────────────────────────────────────

def build_labels(returns: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """
    For each day t, label = best asset to hold on day t+1.
    CASH if ALL next-day ETF returns < 0 OR VIX > 25.
    Returns pd.Series indexed by date (day t), values 0..6 (asset index).
    """
    assets = cfg.ASSETS  # TLT LQD HYG VNQ GLD SLV
    labels = {}

    for i in range(len(returns) - 1):
        today    = returns.index[i]
        tomorrow = returns.index[i + 1]
        next_ret = returns.iloc[i + 1]   # returns ON tomorrow

        # VIX on today
        vix_today = 0.0
        if macro is not None and today in macro.index:
            vix_today = float(macro.loc[today, 'VIX']) if 'VIX' in macro.columns else 0.0

        # CASH if all ETFs negative or high VIX
        etf_rets = [next_ret.get(a, 0.0) for a in assets]
        if all(r < 0 for r in etf_rets) or vix_today > 25:
            labels[today] = cfg.N_ASSETS - 1   # CASH index = 6
        else:
            best_idx = int(np.argmax(etf_rets))
            labels[today] = best_idx

    return pd.Series(labels, name='label')


# ── Dataset ───────────────────────────────────────────────────────────────────

class ETFDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_features_and_labels(
    data: dict,
    embeddings: pd.DataFrame,
    hmm_probs:  pd.DataFrame,
    macro:      pd.DataFrame,
    labels:     pd.Series,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Align all inputs and construct (N, INPUT_DIM) feature matrix.
    """
    assets = cfg.ASSETS

    # Common index: labels define the universe (all days with a valid next-day label)
    common = labels.index
    common = common.intersection(embeddings.dropna().index)
    common = common.intersection(hmm_probs.index)
    common = common.sort_values()

    X_rows, y_rows, dates = [], [], []

    for t in common:
        emb  = embeddings.loc[t].values.astype(np.float32)         # 64
        hmm  = hmm_probs.loc[t].values.astype(np.float32)          # 8

        # Macro (7)
        mac = np.zeros(7, dtype=np.float32)
        if t in macro.index:
            mac = macro.loc[t, ['TNX','DXY','CORP_SPREAD','HY_SPREAD',
                                 'VIX','T10Y2Y','TBILL_3M']].values.astype(np.float32)

        # Lookback returns (LOOKBACK × _N_ETF = 30)
        t_pos = data['returns'].index.get_loc(t) if t in data['returns'].index else -1
        if t_pos < LOOKBACK:
            lb = np.zeros(LOOKBACK * _N_ETF, dtype=np.float32)
        else:
            lb_slice = data['returns'].iloc[t_pos - LOOKBACK:t_pos][assets].values
            lb = lb_slice.flatten().astype(np.float32)

        feat = np.concatenate([emb, hmm, mac, lb])
        assert feat.shape[0] == INPUT_DIM, f"Feature dim mismatch: {feat.shape[0]} != {INPUT_DIM}"

        X_rows.append(feat)
        y_rows.append(int(labels.loc[t]))
        dates.append(t)

    return np.array(X_rows), np.array(y_rows), dates


# ── Model ─────────────────────────────────────────────────────────────────────

class ETFClassifier(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                 n_classes: int = N_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head     = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) → treat as sequence of length 1
        z = self.proj(x).unsqueeze(1)      # (B, 1, d_model)
        z = self.encoder(z).squeeze(1)     # (B, d_model)
        return self.head(z)                # (B, n_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_classifier(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_acc  = 0.0
    best_state    = None
    patience_ctr  = 0
    log = []

    print(f"[CLF] Training for up to {MAX_EPOCHS} epochs...")
    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_correct, train_total = 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        # Val
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits  = model(X_batch)
                val_loss += criterion(logits, y_batch).item()
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += len(y_batch)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"[CLF] Epoch {epoch:3d} | Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f}")
        log.append({'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"[CLF] Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    model.load_state_dict(best_state)
    print(f"[CLF] Best val acc: {best_val_acc:.3f}")
    return log, best_val_acc


# ── Per-class accuracy ────────────────────────────────────────────────────────

def class_accuracy(model, loader):
    model.eval()
    correct = np.zeros(N_CLASSES)
    total   = np.zeros(N_CLASSES)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch).argmax(dim=-1)
            for c in range(N_CLASSES):
                mask = y_batch == c
                correct[c] += (preds[mask] == c).sum().item()
                total[c]   += mask.sum().item()
    per_class = {}
    for i, asset in enumerate(cfg.ALL_ASSETS):
        if total[i] > 0:
            per_class[asset] = round(float(correct[i] / total[i]), 3)
    return per_class


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg.validate()
    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)
    print(f"\n[CLF] Building ETF classifier (Input={INPUT_DIM}, Classes={N_CLASSES})")

    # 1. Load data
    print("[CLF] Loading data...")
    data = load_all()

    # 2. Load TFT embeddings
    print("[CLF] Loading TFT embeddings...")
    cache_path = _hf_download(cfg.FEATURE_CACHE_PATH, cfg.HF_MODELS_REPO)
    embeddings = pd.read_parquet(cache_path)
    embeddings.index = pd.to_datetime(embeddings.index)
    common = embeddings.index.intersection(data['returns'].index)
    embeddings = embeddings.reindex(common)
    print(f"[CLF] Embeddings: {embeddings.shape}")

    # 3. Load HMM probs
    print("[CLF] Loading HMM regime probs...")
    det_path  = _hf_download('models/regime_detector.pkl', cfg.HF_MODELS_REPO)
    detector  = RegimeDetector.load(det_path)
    hmm_feats = load_or_compute_hmm_features(data)
    X_hmm     = detector.scaler.transform(hmm_feats)
    hmm_probs = pd.DataFrame(
        detector.model.predict_proba(X_hmm.values),
        index=hmm_feats.index,
        columns=list(range(cfg.HMM_N_STATES)),
    ).astype(np.float32)
    hmm_probs = hmm_probs.reindex(data['returns'].index)
    print(f"[CLF] HMM probs: {hmm_probs.shape}")

    # 4. Macro
    macro = data.get('macro')

    # 5. Build labels
    print("[CLF] Building next-day labels...")
    labels = build_labels(data['returns'], macro)
    label_counts = pd.Series(labels).value_counts().sort_index()
    print("[CLF] Label distribution:")
    for idx, cnt in label_counts.items():
        asset = cfg.ALL_ASSETS[idx]
        print(f"  {asset:<6}: {cnt:4d} days ({cnt/len(labels):.1%})")

    # 6. Build feature matrix
    print("[CLF] Building feature matrix...")
    X, y, dates = build_features_and_labels(data, embeddings, hmm_probs, macro, labels)
    print(f"[CLF] Dataset: {X.shape[0]} samples × {X.shape[1]} features")

    # 7. Train/val split (80/20 chronological)
    split = int(len(X) * 0.80)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    print(f"[CLF] Train: {len(X_tr)} | Val: {len(X_va)}")

    train_ds = ETFDataset(X_tr, y_tr)
    val_ds   = ETFDataset(X_va, y_va)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # 8. Train
    model = ETFClassifier().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CLF] Model parameters: {n_params:,}")

    log, best_val_acc = train_classifier(model, train_loader, val_loader)

    # 9. Per-class accuracy
    per_class = class_accuracy(model, val_loader)
    print("[CLF] Per-class val accuracy:")
    for asset, acc in per_class.items():
        print(f"  {asset:<6}: {acc:.1%}")

    # 10. Save
    model_path = os.path.join(cfg.LOCAL_TMP, 'etf_classifier.pt')
    meta_path  = os.path.join(cfg.LOCAL_TMP, 'classifier_meta.json')

    torch.save(model.state_dict(), model_path)
    meta = {
        'input_dim':    INPUT_DIM,
        'd_model':      D_MODEL,
        'n_heads':      N_HEADS,
        'n_layers':     N_LAYERS,
        'n_classes':    N_CLASSES,
        'lookback':     LOOKBACK,
        'best_val_acc': best_val_acc,
        'per_class_acc': per_class,
        'train_size':   len(X_tr),
        'val_size':     len(X_va),
        'trained_date': date.today().isoformat(),
        'training_log': log,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n[CLF] Pushing to HuggingFace...")
    _push(model_path, cfg.HF_MODELS_REPO, 'models/etf_classifier.pt')
    _push(meta_path,  cfg.HF_MODELS_REPO, 'models/classifier_meta.json')

    print(f"\n[CLF] ✅ Complete")
    print(f"      Best val accuracy: {best_val_acc:.1%}")
    print(f"      Input dim:         {INPUT_DIM}")
    print(f"      Pushed to:         {cfg.HF_MODELS_REPO}")


if __name__ == '__main__':
    main()

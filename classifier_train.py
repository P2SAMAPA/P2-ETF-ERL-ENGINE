# classifier_train.py — Direct ETF Classification (Option B)
# Predicts single best ETF for next trading day.
# Label = argmax(next-day return). CASH if all ETFs negative or VIX>25.
# Uses focal loss + inverse-frequency class weights to prevent collapse.

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import date
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from loader import load_all
from features import load_or_compute_hmm_features, FeatureScaler
from hmm_train import RegimeDetector

DEVICE     = torch.device('cpu')
_CPU_MODE  = os.environ.get('REALM_CPU_MODE', '0') == '1'
MAX_EPOCHS = 40 if _CPU_MODE else 100
LR         = 3e-4
BATCH_SIZE = 128
LOOKBACK   = 5
N_CLASSES  = cfg.N_ASSETS          # 7
_N_ETF     = len(cfg.ASSETS)       # 6 (no CASH)
INPUT_DIM  = cfg.TFT_EMBEDDING_DIM + cfg.HMM_N_STATES + 7 + (LOOKBACK * _N_ETF)  # 109
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 1
DROPOUT    = 0.3
PATIENCE   = 10
FOCAL_GAMMA= 1.0


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Labels ────────────────────────────────────────────────────────────────────

def build_labels(returns: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """Label = index of best ETF next day. CASH(6) if all negative or VIX>25."""
    assets = cfg.ASSETS
    labels = {}
    for i in range(len(returns) - 1):
        today   = returns.index[i]
        nxt     = returns.iloc[i + 1]
        vix     = float(macro.loc[today, 'VIX']) if (macro is not None and today in macro.index and 'VIX' in macro.columns) else 0.0
        rets    = [float(nxt.get(a, 0.0)) for a in assets]
        if all(r < 0 for r in rets) or vix > 25:
            labels[today] = N_CLASSES - 1   # CASH
        else:
            labels[today] = int(np.argmax(rets))
    return pd.Series(labels, name='label')


# ── Features ──────────────────────────────────────────────────────────────────

def build_feature_matrix(data, embeddings, hmm_probs, macro, labels):
    """Build aligned (N, INPUT_DIM) feature matrix + label array."""
    assets  = cfg.ASSETS
    mac_cols= ['TNX','DXY','CORP_SPREAD','HY_SPREAD','VIX','T10Y2Y','TBILL_3M']
    common  = labels.index
    common  = common.intersection(embeddings.dropna().index)
    common  = common.intersection(hmm_probs.index)
    common  = common.sort_values()

    X, y, dates = [], [], []
    ret_idx = data['returns'].index

    for t in common:
        emb = embeddings.loc[t].values.astype(np.float32)
        hmm = hmm_probs.loc[t].values.astype(np.float32)
        mac = macro.loc[t, mac_cols].values.astype(np.float32) if (macro is not None and t in macro.index) else np.zeros(7, np.float32)

        pos = ret_idx.get_loc(t) if t in ret_idx else -1
        if pos >= LOOKBACK:
            lb = data['returns'].iloc[pos-LOOKBACK:pos][assets].values.flatten().astype(np.float32)
        else:
            lb = np.zeros(LOOKBACK * _N_ETF, np.float32)

        feat = np.concatenate([emb, hmm, mac, lb])
        assert feat.shape[0] == INPUT_DIM, f"dim={feat.shape[0]}"
        X.append(feat)
        y.append(int(labels.loc[t]))
        dates.append(t)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), dates


# ── Dataset ───────────────────────────────────────────────────────────────────

class ETFDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Model ─────────────────────────────────────────────────────────────────────

class ETFClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS,
                 n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.proj    = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        enc_layer    = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):
        z = self.proj(x).unsqueeze(1)
        z = self.encoder(z).squeeze(1)
        return self.head(z)

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=-1)


# ── Focal Loss ────────────────────────────────────────────────────────────────

def focal_loss(logits, targets, class_weights, gamma=FOCAL_GAMMA):
    """Focal loss with class weights. Penalises easy predictions."""
    ce  = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
    pt  = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, class_weights):
    cw        = torch.tensor(class_weights, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    n_steps   = MAX_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=n_steps, pct_start=0.1
    )
    step_sched = True  # step per batch not per epoch

    best_acc, best_state, patience_ctr, log = 0.0, None, 0, []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_correct = tr_total = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(X_b)
            loss   = focal_loss(logits, y_b, cw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tr_correct += (logits.argmax(-1) == y_b).sum().item()
            tr_total   += len(y_b)

        model.eval()
        va_correct = va_total = va_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                logits   = model(X_b)
                va_loss += focal_loss(logits, y_b, cw).item()
                va_correct += (logits.argmax(-1) == y_b).sum().item()
                va_total   += len(y_b)

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        print(f"[CLF] Epoch {epoch:3d} | Train: {tr_acc:.3f} | Val: {va_acc:.3f}")
        log.append({'epoch': epoch, 'train_acc': tr_acc, 'val_acc': va_acc})

        if va_acc > best_acc:
            best_acc, best_state, patience_ctr = va_acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"[CLF] Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return log, best_acc


def per_class_acc(model, loader):
    model.eval()
    correct = np.zeros(N_CLASSES)
    total   = np.zeros(N_CLASSES)
    with torch.no_grad():
        for X_b, y_b in loader:
            preds = model(X_b).argmax(-1)
            for c in range(N_CLASSES):
                m = y_b == c
                correct[c] += (preds[m] == c).sum().item()
                total[c]   += m.sum().item()
    return {cfg.ALL_ASSETS[i]: round(float(correct[i]/max(total[i],1)), 3) for i in range(N_CLASSES)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg.validate()
    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)
    print(f"\n[CLF] ETF Classifier | Input={INPUT_DIM} | Classes={N_CLASSES} | Device={DEVICE}")

    data = load_all()

    print("[CLF] Loading TFT embeddings...")
    emb_path   = _hf_download(cfg.FEATURE_CACHE_PATH, cfg.HF_MODELS_REPO)
    embeddings = pd.read_parquet(emb_path)
    embeddings.index = pd.to_datetime(embeddings.index)
    embeddings = embeddings.reindex(embeddings.index.intersection(data['returns'].index))

    print("[CLF] Loading HMM probs...")
    det_path  = _hf_download('models/regime_detector.pkl', cfg.HF_MODELS_REPO)
    detector  = RegimeDetector.load(det_path)
    hmm_feats = load_or_compute_hmm_features(data)
    X_hmm     = detector.scaler.transform(hmm_feats)
    hmm_probs = pd.DataFrame(
        detector.model.predict_proba(X_hmm.values),
        index=hmm_feats.index, columns=list(range(cfg.HMM_N_STATES))
    ).reindex(data['returns'].index)

    macro = data.get('macro')

    print("[CLF] Building labels...")
    labels = build_labels(data['returns'], macro)
    counts = pd.Series(labels).value_counts().sort_index()
    print("[CLF] Label distribution:")
    for idx, cnt in counts.items():
        print(f"  {cfg.ALL_ASSETS[idx]:<6}: {cnt:4d} days ({cnt/len(labels):.1%})")

    print("[CLF] Building features...")
    X, y, dates = build_feature_matrix(data, embeddings, hmm_probs, macro, labels)
    print(f"[CLF] Dataset: {len(X)} × {X.shape[1]}")

    # Chronological 80/20 split
    split = int(len(X) * 0.80)
    tr_ds = ETFDataset(X[:split], y[:split])
    va_ds = ETFDataset(X[split:], y[split:])
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[CLF] Train: {len(tr_ds)} | Val: {len(va_ds)}")

    # Inverse-frequency class weights
    counts_tr  = np.bincount(y[:split], minlength=N_CLASSES).astype(float)
    counts_tr  = np.maximum(counts_tr, 1)
    class_w    = (len(y[:split]) / (N_CLASSES * counts_tr)).astype(np.float32)
    class_w    = np.clip(class_w, 0.5, 3.0)   # cap weights — prevent rare class dominating
    print(f"[CLF] Class weights: { {cfg.ALL_ASSETS[i]: round(float(class_w[i]),2) for i in range(N_CLASSES)} }")

    model   = ETFClassifier().to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CLF] Parameters: {n_param:,}")

    log, best_acc = train(model, tr_ld, va_ld, class_w)

    pc = per_class_acc(model, va_ld)
    print("[CLF] Per-class val accuracy:")
    for asset, acc in pc.items():
        print(f"  {asset:<6}: {acc:.1%}")

    # Save
    model_path = os.path.join(cfg.LOCAL_TMP, 'etf_classifier.pt')
    meta_path  = os.path.join(cfg.LOCAL_TMP, 'classifier_meta.json')
    torch.save(model.state_dict(), model_path)
    meta = {
        'input_dim': INPUT_DIM, 'd_model': D_MODEL, 'n_heads': N_HEADS,
        'n_layers': N_LAYERS, 'n_classes': N_CLASSES, 'lookback': LOOKBACK,
        'best_val_acc': best_acc, 'per_class_acc': pc,
        'train_size': len(tr_ds), 'val_size': len(va_ds),
        'trained_date': date.today().isoformat(), 'training_log': log,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n[CLF] Pushing to HuggingFace...")
    _push(model_path, cfg.HF_MODELS_REPO, 'models/etf_classifier.pt')
    _push(meta_path,  cfg.HF_MODELS_REPO, 'models/classifier_meta.json')
    print(f"\n[CLF] ✅ Complete | Best val acc: {best_acc:.1%}")


if __name__ == '__main__':
    main()

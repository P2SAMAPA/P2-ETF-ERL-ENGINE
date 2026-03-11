# tft_train.py — Temporal Fusion Transformer for P2-ETF-ERL-ENGINE
# Trains a TFT encoder to produce 64-dim regime embeddings from price
# and macro features. The embedding captures temporal patterns the HMM
# cannot — multi-scale attention across daily and weekly dynamics.
#
# Run on Kaggle GPU (T4/P100 recommended):
#   python tft_train.py
#
# Outputs pushed to HF_MODELS_REPO:
#   models/tft_model.pt
#   models/tft_scaler.pkl
#   data/feature_cache.parquet   ← pre-computed embeddings for all days

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from huggingface_hub import HfApi
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from loader import load_all
from features import (
    compute_tft_features,
    compute_hmm_features,
    FeatureScaler,
)
from hmm_train import RegimeDetector

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[TFT] Using device: {DEVICE}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class TFTDataset(Dataset):
    """
    Sliding window dataset for TFT training.
    Each sample: (context_window, target_return)
    context_window: (H, n_features)
    target_return:  (N_ASSETS,) — next day returns for supervised pretraining
    """

    def __init__(
        self,
        features: pd.DataFrame,
        returns:  pd.DataFrame,
        context_len: int = cfg.TFT_CONTEXT_LENGTH,
    ):
        self.features    = features.values.astype(np.float32)
        self.feat_index  = features.index
        self.context_len = context_len

        # Align returns to feature index
        aligned_returns  = returns.reindex(features.index).fillna(0.0)
        self.returns     = aligned_returns.values.astype(np.float32)
        self.n_samples   = len(features) - context_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.context_len]        # (H, F)
        y = self.returns[idx + self.context_len]               # (N_ASSETS,)
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ── TFT Architecture ───────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network — core building block of TFT.
    Enables selective information flow with gating.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = cfg.TFT_DROPOUT):
        super().__init__()
        self.fc1     = nn.Linear(input_dim,  hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, output_dim)
        self.gate    = nn.Linear(hidden_dim, output_dim)
        self.proj    = nn.Linear(input_dim,  output_dim) \
                       if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(output_dim)
        self.elu     = nn.ELU()

    def forward(self, x):
        residual = self.proj(x)
        h        = self.elu(self.fc1(x))
        h        = self.dropout(h)
        out      = self.fc2(h)
        gate     = torch.sigmoid(self.gate(h))
        return self.norm(gate * out + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network — learns which features matter most.
    Produces soft feature weights via softmax.
    """

    def __init__(self, n_features: int, hidden_dim: int,
                 dropout: float = cfg.TFT_DROPOUT):
        super().__init__()
        self.grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
            for _ in range(n_features)
        ])
        self.softmax_weights = nn.Sequential(
            nn.Linear(n_features * hidden_dim, n_features),
            nn.Softmax(dim=-1),
        )
        self.hidden_dim  = hidden_dim
        self.n_features  = n_features

    def forward(self, x):
        # x: (B, T, F) or (B, F)
        processed = []
        for i, grn in enumerate(self.grns):
            xi = x[..., i:i+1]
            processed.append(grn(xi))

        stacked  = torch.stack(processed, dim=-2)    # (B, T, F, H)
        flat     = stacked.flatten(start_dim=-2)      # (B, T, F*H)
        weights  = self.softmax_weights(flat)         # (B, T, F)
        weights  = weights.unsqueeze(-1)              # (B, T, F, 1)
        selected = (stacked * weights).sum(dim=-2)    # (B, T, H)
        return selected, weights.squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for regime embedding generation.

    Architecture:
        Input features
          → Variable Selection Network (learns which features matter)
          → LSTM encoder (captures temporal dynamics)
          → Multi-head self-attention (captures long-range dependencies)
          → Gated skip connection
          → Linear projection → 64-dim embedding
          → Output head → next-day returns (for training supervision)

    At inference: we use the 64-dim embedding, not the return prediction.
    """

    def __init__(self, n_features: int):
        super().__init__()

        H  = cfg.TFT_HIDDEN_SIZE        # 64
        Hc = cfg.TFT_HIDDEN_CONT_SIZE   # 32
        D  = cfg.TFT_DROPOUT

        # Variable selection
        self.vsn = VariableSelectionNetwork(n_features, H, D)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size  = H,
            hidden_size = H,
            num_layers  = cfg.TFT_LSTM_LAYERS,
            dropout     = D if cfg.TFT_LSTM_LAYERS > 1 else 0,
            batch_first = True,
        )
        self.lstm_gate = GatedResidualNetwork(H, H, H, D)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim   = H,
            num_heads   = cfg.TFT_ATTENTION_HEADS,
            dropout     = D,
            batch_first = True,
        )
        self.attn_gate = GatedResidualNetwork(H, H, H, D)
        self.attn_norm = nn.LayerNorm(H)

        # Positionwise feed-forward
        self.ff   = GatedResidualNetwork(H, H * 2, H, D)

        # Embedding projection (class token style — take last timestep)
        self.embed_proj = nn.Sequential(
            nn.Linear(H, cfg.TFT_EMBEDDING_DIM),
            nn.LayerNorm(cfg.TFT_EMBEDDING_DIM),
        )

        # Output head — predicts next-day returns for N_ASSETS (no CASH)
        self.output_head = nn.Sequential(
            nn.Linear(cfg.TFT_EMBEDDING_DIM, Hc),
            nn.ReLU(),
            nn.Dropout(D),
            nn.Linear(Hc, len(cfg.ASSETS)),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor shape (B, T, F)

        Returns
        -------
        embedding    : (B, TFT_EMBEDDING_DIM)
        pred_returns : (B, N_ASSETS)
        attn_weights : (B, T, T)
        """
        # Variable selection
        selected, feat_weights = self.vsn(x)           # (B, T, H)

        # LSTM
        lstm_out, _ = self.lstm(selected)              # (B, T, H)
        lstm_out    = self.lstm_gate(lstm_out)

        # Multi-head attention
        attn_out, attn_weights = self.attn(
            lstm_out, lstm_out, lstm_out
        )
        attn_out = self.attn_gate(attn_out)
        attn_out = self.attn_norm(attn_out + lstm_out)

        # Feed-forward
        ff_out = self.ff(attn_out)                     # (B, T, H)

        # Take last timestep as embedding
        last    = ff_out[:, -1, :]                     # (B, H)
        embedding = self.embed_proj(last)              # (B, 64)

        # Predict next-day returns
        pred_returns = self.output_head(embedding)     # (B, N_ASSETS)

        return embedding, pred_returns, attn_weights


# ── Training ───────────────────────────────────────────────────────────────────

def train_tft(
    model:       TemporalFusionTransformer,
    train_loader: DataLoader,
    val_loader:  DataLoader,
) -> dict:
    """
    Train TFT with early stopping.
    Loss: MSE on next-day returns (supervised pretraining for embedding quality).

    Returns
    -------
    dict: training log with best_epoch, best_val_loss
    """
    optimizer = optim.Adam(model.parameters(), lr=cfg.TFT_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )
    criterion = nn.MSELoss()

    best_val_loss  = np.inf
    best_state     = None
    patience_count = 0
    train_log      = []

    print(f"\n[TFT] Training for up to {cfg.TFT_MAX_EPOCHS} epochs...")
    print(f"[TFT] Train batches: {len(train_loader)} | "
          f"Val batches: {len(val_loader)}")

    for epoch in range(1, cfg.TFT_MAX_EPOCHS + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            _, pred_returns, _ = model(x_batch)
            loss = criterion(pred_returns, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                _, pred_returns, _ = model(x_batch)
                loss = criterion(pred_returns, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)

        train_log.append({
            'epoch':      epoch,
            'train_loss': float(train_loss),
            'val_loss':   float(val_loss),
        })

        print(f"[TFT] Epoch {epoch:3d} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.TFT_EARLY_STOP_PAT:
                print(f"[TFT] Early stop at epoch {epoch} "
                      f"(patience={cfg.TFT_EARLY_STOP_PAT})")
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
        print(f"[TFT] Restored best weights (val_loss={best_val_loss:.6f})")

    return {
        'best_val_loss': float(best_val_loss),
        'best_epoch':    int(np.argmin([r['val_loss'] for r in train_log]) + 1),
        'log':           train_log,
    }


# ── Embedding Extraction ───────────────────────────────────────────────────────

def extract_embeddings(
    model:    TemporalFusionTransformer,
    features: pd.DataFrame,
    context_len: int = cfg.TFT_CONTEXT_LENGTH,
) -> pd.DataFrame:
    """
    Extract 64-dim TFT embeddings for every day in features.
    Uses a sliding window — first context_len days have no embedding
    (filled with the first available embedding).

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex
        Columns = [emb_0, emb_1, ..., emb_63]
    """
    model.eval()
    X = features.values.astype(np.float32)
    embeddings = []

    batch_size = 256
    indices    = list(range(context_len, len(X)))

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_x   = np.stack([
                X[i - context_len : i] for i in batch_idx
            ])
            batch_x   = torch.FloatTensor(batch_x).to(DEVICE)
            emb, _, _ = model(batch_x)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings)   # (T - context_len, 64)

    # Pad first context_len rows with first available embedding
    first_emb  = embeddings[0:1]
    padding    = np.repeat(first_emb, context_len, axis=0)
    embeddings = np.vstack([padding, embeddings])

    cols = [f'emb_{i}' for i in range(cfg.TFT_EMBEDDING_DIM)]
    df   = pd.DataFrame(embeddings, index=features.index, columns=cols)

    print(f"[TFT] Embeddings extracted: {df.shape}")
    return df


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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg.validate()
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[TFT] Loading data...")
    data = load_all()

    # ── 2. Load HMM regime labels ─────────────────────────────────────────────
    print("\n[TFT] Loading HMM regime labels...")
    try:
        from huggingface_hub import hf_hub_download
        det_path = hf_hub_download(
            repo_id     = cfg.HF_MODELS_REPO,
            filename    = "models/regime_detector.pkl",
            repo_type   = "dataset",
            token       = cfg.HF_TOKEN,
            force_download = True,
        )
        detector = RegimeDetector.load(det_path)

        hmm_feats = compute_hmm_features(data['prices'], data['benchmark'])
        scaler    = detector.scaler
        X_all     = scaler.transform(hmm_feats).values
        labels    = pd.Series(
            detector.model.predict(X_all),
            index = hmm_feats.index,
            name  = 'regime',
        )
        print(f"[TFT] Regime labels loaded: {len(labels)} days")
    except Exception as e:
        print(f"[TFT] Could not load HMM labels ({e}) — "
              f"training without regime feature")
        labels = None

    # ── 3. Compute TFT features ───────────────────────────────────────────────
    print("\n[TFT] Computing features...")
    tft_feats = compute_tft_features(
        data['prices'], data['benchmark'], labels
    )
    n_features = tft_feats.shape[1]
    print(f"[TFT] Feature matrix: {tft_feats.shape}")

    # ── 4. Scale features ─────────────────────────────────────────────────────
    scaler_tft  = FeatureScaler()
    train_feats = tft_feats[tft_feats.index <= cfg.TRAIN_END]
    scaler_tft.fit(train_feats)
    tft_scaled  = scaler_tft.transform(tft_feats)

    # ── 5. Build train / val splits ───────────────────────────────────────────
    # Val = last 2 years of training data
    val_start   = '2023-01-01'
    train_scaled = tft_scaled[tft_scaled.index < val_start]
    val_scaled   = tft_scaled[
        (tft_scaled.index >= val_start) &
        (tft_scaled.index <= cfg.TRAIN_END)
    ]

    # Get aligned returns
    train_ret = data['returns'].reindex(train_scaled.index).fillna(0.0)
    val_ret   = data['returns'].reindex(val_scaled.index).fillna(0.0)

    train_dataset = TFTDataset(train_scaled, train_ret)
    val_dataset   = TFTDataset(val_scaled,   val_ret)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.TFT_BATCH_SIZE,
        shuffle     = True,
        num_workers = 2,
        pin_memory  = DEVICE.type == 'cuda',
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.TFT_BATCH_SIZE,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = DEVICE.type == 'cuda',
    )

    print(f"\n[TFT] Train samples: {len(train_dataset)} | "
          f"Val samples: {len(val_dataset)}")

    # ── 6. Build and train model ──────────────────────────────────────────────
    print(f"\n[TFT] Building model (n_features={n_features})...")
    model = TemporalFusionTransformer(n_features=n_features).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TFT] Trainable parameters: {n_params:,}")

    train_log = train_tft(model, train_loader, val_loader)

    print(f"\n[TFT] Best epoch: {train_log['best_epoch']} | "
          f"Best val loss: {train_log['best_val_loss']:.6f}")

    # ── 7. Extract embeddings for all days ────────────────────────────────────
    print("\n[TFT] Extracting embeddings for all days...")
    embeddings_df = extract_embeddings(model, tft_scaled)

    # ── 8. Build feature cache (embeddings + regime probs) ───────────────────
    print("\n[TFT] Building feature cache...")
    feature_cache = embeddings_df.copy()
    feature_cache.index.name = 'date'

    # ── 9. Save locally ───────────────────────────────────────────────────────
    print("\n[TFT] Saving outputs...")

    model_path  = os.path.join(cfg.LOCAL_TMP, "tft_model.pt")
    scaler_path = os.path.join(cfg.LOCAL_TMP, "tft_scaler.pkl")
    cache_path  = os.path.join(cfg.LOCAL_TMP, "feature_cache.parquet")
    log_path    = os.path.join(cfg.LOCAL_TMP, "tft_training_log.json")
    meta_path   = os.path.join(cfg.LOCAL_TMP, "tft_meta.json")

    # Save model with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features':       n_features,
        'feature_names':    list(tft_feats.columns),
        'trained_at':       datetime.utcnow().isoformat(),
        'train_log':        train_log,
    }, model_path)
    print(f"[TFT] Model saved → {model_path}")

    scaler_tft.save(scaler_path)
    feature_cache.to_parquet(cache_path)
    print(f"[TFT] Feature cache saved → {cache_path} "
          f"({len(feature_cache)} rows)")

    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)

    meta = {
        'n_features':    n_features,
        'embedding_dim': cfg.TFT_EMBEDDING_DIM,
        'context_len':   cfg.TFT_CONTEXT_LENGTH,
        'feature_names': list(tft_feats.columns),
        'trained_at':    datetime.utcnow().isoformat(),
        'best_epoch':    train_log['best_epoch'],
        'best_val_loss': train_log['best_val_loss'],
        'n_params':      n_params,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # ── 10. Push to HuggingFace ───────────────────────────────────────────────
    print("\n[TFT] Pushing to HuggingFace...")

    push_to_hf(model_path,  cfg.HF_MODELS_REPO, cfg.TFT_MODEL_PATH)
    push_to_hf(scaler_path, cfg.HF_MODELS_REPO, cfg.TFT_SCALER_PATH)
    push_to_hf(meta_path,   cfg.HF_MODELS_REPO, "models/tft_meta.json")
    push_to_hf(log_path,    cfg.HF_MODELS_REPO, "models/tft_training_log.json")
    push_to_hf(cache_path,  cfg.HF_MODELS_REPO, cfg.FEATURE_CACHE_PATH)

    print("\n[TFT] ✅ Complete")
    print(f"      Embedding dim:  {cfg.TFT_EMBEDDING_DIM}")
    print(f"      Feature cache:  {len(feature_cache)} days")
    print(f"      Pushed to:      {cfg.HF_MODELS_REPO}")


if __name__ == "__main__":
    main()

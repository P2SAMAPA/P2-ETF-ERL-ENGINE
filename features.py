# features.py — Feature engineering for P2-ETF-ERL-ENGINE
# Computes price-derived features per asset and macro cross-asset features
# Output feeds into: HMM regime detector, TFT encoder, DDPG state vector

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


# ── Price Features (per asset) ─────────────────────────────────────────────────

def daily_return(prices: pd.Series) -> pd.Series:
    """F1: Daily percentage return."""
    return prices.pct_change()


def rolling_vol(prices: pd.Series, window: int = cfg.SHORT_WINDOW) -> pd.Series:
    """F2: Rolling N-day return volatility."""
    returns = prices.pct_change()
    return returns.rolling(window).std()


def vol_ratio(prices: pd.Series,
              short: int = cfg.SHORT_WINDOW,
              long:  int = cfg.MED_WINDOW) -> pd.Series:
    """F3: Short vol / long vol ratio — regime indicator."""
    r   = prices.pct_change()
    sv  = r.rolling(short).std()
    lv  = r.rolling(long).std()
    return sv / (lv + 1e-8)


def momentum(prices: pd.Series, window: int = cfg.MED_WINDOW) -> pd.Series:
    """F4/F5: N-day price momentum."""
    return prices.pct_change(window)


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """F6: Relative Strength Index (0-100)."""
    delta  = prices.diff()
    gain   = delta.clip(lower=0).rolling(window).mean()
    loss   = (-delta.clip(upper=0)).rolling(window).mean()
    rs     = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all price-derived features for every asset.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns = ASSETS, Index = DatetimeIndex

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (asset, feature)
        e.g. ('TLT', 'daily_return'), ('TLT', 'rolling_5d_vol'), ...
        Shape: (T, N_ASSETS * N_PRICE_FEATURES)
    """
    frames = {}
    for asset in cfg.ASSETS:
        p = prices[asset]
        frames[(asset, 'daily_return')]   = daily_return(p)
        frames[(asset, 'rolling_5d_vol')] = rolling_vol(p, cfg.SHORT_WINDOW)
        frames[(asset, 'vol_ratio')]      = vol_ratio(p, cfg.SHORT_WINDOW, cfg.MED_WINDOW)
        frames[(asset, 'momentum_20d')]   = momentum(p, cfg.MED_WINDOW)
        frames[(asset, 'momentum_60d')]   = momentum(p, cfg.LONG_WINDOW)
        frames[(asset, 'rsi_14')]         = rsi(p, 14)

    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['asset', 'feature'])
    return df


# ── Macro Features (cross-asset, one value per day) ───────────────────────────

def compute_macro_features(prices: pd.DataFrame,
                           benchmark: pd.Series) -> pd.DataFrame:
    """
    Compute macro/cross-asset features that capture regime dynamics.

    These are the primary inputs to the HMM regime detector.

    Parameters
    ----------
    prices    : pd.DataFrame — ETF prices (ASSETS columns)
    benchmark : pd.Series   — AGG prices

    Returns
    -------
    pd.DataFrame
        Columns = MACRO_FEATURES, Index = DatetimeIndex
    """
    ret = prices.pct_change()
    b_ret = benchmark.pct_change()

    macro = pd.DataFrame(index=prices.index)

    # Yield curve slope proxy: TLT return - HYG return
    # Rising → flight to duration, falling → risk appetite
    macro['yield_curve_slope'] = (
        ret['TLT'].rolling(cfg.MED_WINDOW).mean() -
        ret['HYG'].rolling(cfg.MED_WINDOW).mean()
    )

    # Credit spread proxy: LQD return - HYG return
    # Positive → IG outperforming HY → credit stress widening
    macro['credit_spread'] = (
        ret['LQD'].rolling(cfg.MED_WINDOW).mean() -
        ret['HYG'].rolling(cfg.MED_WINDOW).mean()
    )

    # Vol regime: rolling 20d vol of benchmark returns
    # High → risk-off, low → complacency
    macro['vol_regime'] = b_ret.rolling(cfg.MED_WINDOW).std()

    # Real rate direction proxy: TLT 60d momentum
    # Positive → rates falling (TLT rising), negative → rates rising
    macro['real_rate_direction'] = ret['TLT'].rolling(cfg.LONG_WINDOW).mean()

    # Risk appetite: GLD vs SLV relative momentum
    # GLD outperforming SLV → safe haven demand (risk-off)
    # SLV outperforming GLD → industrial demand (risk-on)
    macro['risk_appetite'] = (
        momentum(prices['SLV'], cfg.MED_WINDOW) -
        momentum(prices['GLD'], cfg.MED_WINDOW)
    )

    return macro


# ── HMM Input Features ─────────────────────────────────────────────────────────

def compute_hmm_features(prices: pd.DataFrame,
                         benchmark: pd.Series) -> pd.DataFrame:
    """
    Assemble the feature matrix used to train and run the HMM.
    Uses macro features only — HMM operates on regime-level dynamics,
    not individual asset features.

    Returns
    -------
    pd.DataFrame
        Clean (no NaN), scaled-ready feature matrix for hmmlearn
    """
    macro = compute_macro_features(prices, benchmark)
    macro = macro.dropna()
    return macro


# ── TFT Input Features ─────────────────────────────────────────────────────────

def compute_tft_features(prices: pd.DataFrame,
                         benchmark: pd.Series,
                         regime_labels: pd.Series = None) -> pd.DataFrame:
    """
    Assemble the full feature matrix for the TFT encoder.
    Combines price features (per asset) + macro features + optional regime label.

    Returns
    -------
    pd.DataFrame
        Flat columns, Index = DatetimeIndex
        Shape: (T, N_ASSETS * N_PRICE_FEATURES + N_MACRO_FEATURES [+ 1])
    """
    # Price features — flatten MultiIndex to single level
    price_feats = compute_price_features(prices)
    price_feats.columns = [
        f"{asset}_{feat}"
        for asset, feat in price_feats.columns
    ]

    # Macro features
    macro_feats = compute_macro_features(prices, benchmark)

    # Combine
    combined = price_feats.join(macro_feats, how='inner')

    # Add regime label if available (as a continuous feature, not one-hot)
    if regime_labels is not None:
        combined = combined.join(
            regime_labels.rename('regime_label'),
            how='left'
        )
        combined['regime_label'] = combined['regime_label'].fillna(
            combined['regime_label'].mode()[0]
        )

    combined = combined.dropna()
    print(f"[features] TFT feature matrix: {combined.shape} "
          f"({combined.index[0].date()} → {combined.index[-1].date()})")
    return combined


# ── DDPG State Features ────────────────────────────────────────────────────────

def compute_ddpg_state(
    tft_embedding: np.ndarray,          # (64,) from TFT encoder
    hmm_probs: np.ndarray,              # (8,)  from HMM
    current_weights: np.ndarray,        # (7,)  current portfolio weights
    rolling_sharpe: float,              # scalar
) -> np.ndarray:
    """
    Assemble the 80-dim DDPG state vector from its four components.

    Parameters
    ----------
    tft_embedding   : np.ndarray shape (TFT_EMBEDDING_DIM,)
    hmm_probs       : np.ndarray shape (HMM_N_STATES,)
    current_weights : np.ndarray shape (N_ASSETS,) — must sum to 1
    rolling_sharpe  : float — recent portfolio Sharpe ratio

    Returns
    -------
    np.ndarray shape (DDPG_STATE_DIM,) = (80,)
    """
    state = np.concatenate([
        tft_embedding.flatten(),
        hmm_probs.flatten(),
        current_weights.flatten(),
        np.array([rolling_sharpe]),
    ])
    assert state.shape[0] == cfg.DDPG_STATE_DIM, (
        f"State dim mismatch: got {state.shape[0]}, "
        f"expected {cfg.DDPG_STATE_DIM}"
    )
    return state.astype(np.float32)


# ── Normalisation ──────────────────────────────────────────────────────────────

class FeatureScaler:
    """
    Wraps sklearn StandardScaler with fit/transform/save/load.
    Always fit on training data only — never on test or live data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> 'FeatureScaler':
        self.scaler.fit(df.values)
        self.fitted = True
        self.feature_names = list(df.columns)
        print(f"[scaler] Fitted on {len(df)} rows, "
              f"{len(df.columns)} features")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("[scaler] Must call fit() before transform()")
        scaled = self.scaler.transform(df[self.feature_names].values)
        return pd.DataFrame(scaled,
                            index=df.index,
                            columns=self.feature_names)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) \
            if os.path.dirname(path) else None
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[scaler] Saved → {path}")

    @staticmethod
    def load(path: str) -> 'FeatureScaler':
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"[scaler] Loaded ← {path}")
        return scaler


# ── Rolling Sharpe ─────────────────────────────────────────────────────────────

def rolling_sharpe(
    portfolio_returns: pd.Series,
    window: int = cfg.KELLY_SHARPE_WINDOW,
    rf: float = 0.0,
) -> pd.Series:
    """
    Compute rolling annualised Sharpe ratio.

    Parameters
    ----------
    portfolio_returns : pd.Series of daily returns
    window            : int — rolling window in days
    rf                : float — daily risk-free rate (default 0)

    Returns
    -------
    pd.Series of rolling Sharpe values
    """
    excess  = portfolio_returns - rf / 252
    mean_r  = excess.rolling(window).mean()
    std_r   = excess.rolling(window).std()
    sharpe  = (mean_r / (std_r + 1e-8)) * np.sqrt(252)
    return sharpe


def current_sharpe(
    portfolio_returns: pd.Series,
    window: int = cfg.KELLY_SHARPE_WINDOW,
) -> float:
    """Return the most recent rolling Sharpe value as a scalar."""
    s = rolling_sharpe(portfolio_returns, window)
    val = s.dropna()
    return float(val.iloc[-1]) if len(val) > 0 else 0.0


# ── Smoke Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from loader import load_all

    print("Loading data...")
    data = load_all()

    print("\n── Price Features ────────────────────────────")
    pf = compute_price_features(data['prices'])
    print(f"Shape: {pf.shape}")
    print(pf.tail(2).to_string())

    print("\n── Macro Features ────────────────────────────")
    mf = compute_macro_features(data['prices'], data['benchmark'])
    print(f"Shape: {mf.shape}")
    print(mf.tail(2).to_string())

    print("\n── HMM Features ──────────────────────────────")
    hf = compute_hmm_features(data['prices'], data['benchmark'])
    print(f"Shape: {hf.shape}")
    print(f"NaN count: {hf.isna().sum().sum()}")

    print("\n── TFT Features ──────────────────────────────")
    tf = compute_tft_features(data['prices'], data['benchmark'])
    print(f"Shape: {tf.shape}")
    print(f"NaN count: {tf.isna().sum().sum()}")

    print("\n── Scaler Test ───────────────────────────────")
    scaler = FeatureScaler()
    train_feats = hf[hf.index <= cfg.TRAIN_END]
    live_feats  = hf[hf.index >= cfg.LIVE_START]
    train_scaled = scaler.fit_transform(train_feats)
    live_scaled  = scaler.transform(live_feats)
    print(f"Train scaled mean: {train_scaled.mean().mean():.4f} (should be ~0)")
    print(f"Train scaled std:  {train_scaled.std().mean():.4f} (should be ~1)")
    print(f"Live  scaled mean: {live_scaled.mean().mean():.4f}")

    print("\n── DDPG State Test ───────────────────────────")
    dummy_embedding = np.random.randn(cfg.TFT_EMBEDDING_DIM).astype(np.float32)
    dummy_probs     = np.ones(cfg.HMM_N_STATES) / cfg.HMM_N_STATES
    dummy_weights   = np.ones(cfg.N_ASSETS) / cfg.N_ASSETS
    state = compute_ddpg_state(dummy_embedding, dummy_probs,
                               dummy_weights, 0.5)
    print(f"State shape: {state.shape} (expected {cfg.DDPG_STATE_DIM})")
    print("✅ All feature tests passed")


# ── Macro from source repo (preferred over recomputing) ────────────────────────

def load_or_compute_hmm_features(data: dict) -> pd.DataFrame:
    """
    Use pre-computed macro.parquet if available in data dict,
    otherwise compute from prices. Aligns column names to cfg.MACRO_FEATURES.
    """
    if 'macro' in data and data['macro'] is not None:
        macro = data['macro'].copy()

        # Actual columns in p2-etf-deepwave-dl/data/macro.parquet:
        # TNX, DXY, CORP_SPREAD, HY_SPREAD, VIX, T10Y2Y, TBILL_3M
        # Use all available columns — HMM will learn from whatever is present
        SOURCE_MACRO_COLS = [
            'TNX',         # 10-year Treasury yield
            'DXY',         # USD index
            'CORP_SPREAD', # IG corporate spread
            'HY_SPREAD',   # HY spread
            'VIX',         # Equity vol (regime signal)
            'T10Y2Y',      # Yield curve slope (10y - 2y)
            'TBILL_3M',    # 3-month T-bill rate
        ]
        available = [c for c in SOURCE_MACRO_COLS if c in macro.columns]
        if len(available) >= 3:
            print(f"[features] Using pre-computed macro: {available}")
            return macro[available].dropna()

    # Fallback: compute from prices
    print("[features] Computing macro features from prices...")
    return compute_hmm_features(data['prices'], data['benchmark'])

# config.py — Single source of truth for P2-ETF-ERL-ENGINE
# All hyperparameters, paths, and settings live here.
# Every other file imports from this module.

import os

# ── Repository Identity ────────────────────────────────────────────────────────
PROJECT_NAME    = "P2-ETF-ERL-ENGINE"
VERSION         = "1.0.0"

# ── HuggingFace Repos ──────────────────────────────────────────────────────────
HF_SOURCE_REPO  = os.environ.get("HF_SOURCE_REPO",  "P2SAMAPA/p2-etf-deepwave-dl")
HF_MODELS_REPO  = os.environ.get("HF_MODELS_REPO",  "P2SAMAPA/p2-etf-erl-models")
HF_RESULTS_REPO = os.environ.get("HF_RESULTS_REPO", "P2SAMAPA/p2-etf-erl-results")
HF_TOKEN        = os.environ.get("HF_TOKEN",        None)

# ── API Keys ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY",  None)

# ── Asset Universe ─────────────────────────────────────────────────────────────
# Fixed income ETFs (original set)
FI_ETFS = ['TLT', 'LQD', 'HYG', 'VNQ', 'GLD', 'MBB', 'PFF', 'SLV']

# Equity ETFs (new module)
EQUITY_ETFS = [
    "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLU", "XME", "IWF", "XSD", "XBI", "GDX", "IWM"
]

# Combined list used by the pipeline
ASSETS = FI_ETFS + EQUITY_ETFS

CASH            = 'CASH'
ALL_ASSETS      = ASSETS + [CASH]                # 8 FI + 15 Equity + CASH = 24 total
N_ASSETS        = len(ALL_ASSETS)                # 24
BENCHMARK       = 'AGG'

# ── Data ───────────────────────────────────────────────────────────────────────
DATA_START      = '2008-01-01'
LIVE_START      = '2025-01-01'
TRAIN_END       = '2024-12-31'

# ── Features ───────────────────────────────────────────────────────────────────
PRICE_FEATURES  = [
    'daily_return',        # (p_t / p_{t-1}) - 1
    'rolling_5d_vol',      # std of last 5 daily returns
    'vol_ratio',           # 5d_vol / 20d_vol
    'momentum_20d',        # (p_t / p_{t-20}) - 1
    'momentum_60d',        # (p_t / p_{t-60}) - 1
    'rsi_14',              # 14-day RSI
]

MACRO_FEATURES  = [
    'yield_curve_slope',   # TLT return - HYG return (proxy for 10y-2y)
    'credit_spread',       # LQD return - HYG return (proxy for IG-HY spread)
    'vol_regime',          # rolling 20d vol of benchmark
    'real_rate_direction', # TLT momentum (proxy for real rate trend)
    'risk_appetite',       # GLD vs SLV relative momentum
]

N_PRICE_FEATURES = len(PRICE_FEATURES)           # 6
N_MACRO_FEATURES = len(MACRO_FEATURES)            # 5

# ── Lookback Windows ───────────────────────────────────────────────────────────
H               = 60       # lookback days for TFT and DDPG state
SHORT_WINDOW    = 5
MED_WINDOW      = 20
LONG_WINDOW     = 60

# ── HMM Regime Detector ────────────────────────────────────────────────────────
HMM_N_STATES    = 8
HMM_N_ITER      = 200
HMM_TOL         = 1e-4
HMM_COVARIANCE  = 'full'

REGIME_NAMES    = {
    0: 'Mid Cycle Growth',
    1: 'Late Cycle',
    2: 'Curve Flattening',
    3: 'Credit Stress',
    4: 'Risk Off',
    5: 'Low Vol Expansion',
    6: 'Acute Crisis',
    7: 'Recovery',
}

# Which regimes each ensemble policy specialises in
POLICY_A_REGIMES = [4, 6, 7]                     # crisis — Credit Stress, Acute Crisis, Risk Off
POLICY_B_REGIMES = [0, 1, 5]                     # expansion — Recovery, Mid Cycle, Low Vol Expansion
POLICY_C_REGIMES = list(range(HMM_N_STATES))        # all regimes

# ── Temporal Fusion Transformer ────────────────────────────────────────────────
TFT_HIDDEN_SIZE         = 64
TFT_ATTENTION_HEADS     = 4
TFT_DROPOUT             = 0.1
TFT_HIDDEN_CONT_SIZE    = 32
TFT_LSTM_LAYERS         = 2
TFT_CONTEXT_LENGTH      = H
TFT_PREDICTION_LENGTH   = 1
TFT_EMBEDDING_DIM       = 64
# ── CPU mode (GitHub Actions) — reduced epochs to fit 6hr limit ───────────
import os as _os
_CPU_MODE = _os.environ.get('REALM_CPU_MODE', '0') == '1'

TFT_MAX_EPOCHS          = 10 if _CPU_MODE else 30
TFT_BATCH_SIZE          = 64
TFT_LR                  = 1e-3
TFT_EARLY_STOP_PAT      = 5

# ── DDPG ───────────────────────────────────────────────────────────────────────
# State: TFT embedding (64) + HMM probs (8) + weights (19) + rolling sharpe (1)
DDPG_STATE_DIM          = TFT_EMBEDDING_DIM + HMM_N_STATES + N_ASSETS + 1  # 64+8+19+1=92
DDPG_ACTION_DIM         = N_ASSETS           # 19
DDPG_ACTOR_HIDDEN       = [256, 128]
DDPG_CRITIC_HIDDEN      = [256, 128]
DDPG_LR_ACTOR           = 1e-4
DDPG_LR_CRITIC          = 1e-3
DDPG_GAMMA              = 0.99
DDPG_TAU                = 0.005
DDPG_BUFFER_SIZE        = 10_000
DDPG_BATCH_SIZE         = 64
DDPG_MAX_EPOCHS         = 30 if _CPU_MODE else 100
DDPG_EARLY_STOP_PAT     = 15
DDPG_OU_THETA           = 0.15
DDPG_OU_SIGMA           = 0.2
TRANSACTION_COST        = 0.001

# ── ERL ────────────────────────────────────────────────────────────────────────
ERL_REWARD_THRESHOLD    = 0.0        # reflect only when r(1) < threshold
ERL_MIN_EXCESS_TO_STORE = 0.005      # 50bps gate to store reflection
ERL_MAX_RULES           = 20         # rolling rulebook size
ERL_REFLECTION_PROVIDER = "gemini"   # "gemini" | "rule_based"
ERL_GEMINI_MODEL        = "gemini-2.0-flash"
ERL_MAX_REFLECTION_TOKENS = 512
ERL_SECOND_ATTEMPT_EPISODES = 5

# ── Ensemble Gating ────────────────────────────────────────────────────────────
ENSEMBLE_MIN_AGREEMENT    = 0.6
ENSEMBLE_CRISIS_THRESHOLD = 0.4      # P(crisis regime) > this → Policy A leads

# ── Kelly Position Sizing ──────────────────────────────────────────────────────
KELLY_BASE_FRACTION     = 0.25
KELLY_MIN_FRACTION      = 0.05
KELLY_MAX_FRACTION      = 0.40
KELLY_SHARPE_WINDOW     = 20
KELLY_MAX_CASH          = 0.30

# ── Portfolio Simulation ───────────────────────────────────────────────────────
INITIAL_CAPITAL         = 10_000.0
HOLD_DAYS               = 1

# ── Scoring & Consensus ────────────────────────────────────────────────────────
SCORE_WINDOW            = 60
MIN_DAYS_PROVISIONAL    = 5
MIN_DAYS_MODERATE       = 15
MIN_DAYS_FULL           = 30
SCORE_EXCESS_WEIGHT     = 0.50
SCORE_SHARPE_WEIGHT     = 0.30
SCORE_DRAWDOWN_WEIGHT   = 0.20

# ── HF File Paths ─────────────────────────────────────────────────────────────
# Models repo
HMM_MODEL_PATH          = "models/hmm_model.pkl"
TFT_MODEL_PATH          = "models/tft_model.pt"
POLICY_A_PATH           = "models/policy_A.pt"
POLICY_B_PATH           = "models/policy_B.pt"
POLICY_C_PATH           = "models/policy_C.pt"
FEATURE_CACHE_PATH      = "data/feature_cache.parquet"
REGIME_SCALER_PATH      = "models/regime_scaler.pkl"
TFT_SCALER_PATH         = "models/tft_scaler.pkl"

# Results repo
LATEST_SIGNAL_PATH      = "results/latest_signal.json"
SIGNAL_HISTORY_PATH     = "results/signal_history.json"
REGIME_HISTORY_PATH     = "results/regime_history.csv"
RULEBOOK_PATH           = "results/rulebook.json"
PERFORMANCE_PATH        = "results/performance_summary.json"
ENSEMBLE_HISTORY_PATH   = "results/ensemble_history.csv"

# ── Local Paths ────────────────────────────────────────────────────────────────
LOCAL_TMP               = "/tmp/realm"
KAGGLE_WORKING          = "/kaggle/working"
KAGGLE_REPO_PATH        = "/kaggle/working/repo"

# ── Misc ───────────────────────────────────────────────────────────────────────
MAX_HISTORY_RECORDS     = 90
RANDOM_SEED             = 42


# ── Validation ─────────────────────────────────────────────────────────────────
def validate():
    """
    Sanity check — call at the top of every training script
    to confirm all required environment variables are present.
    """
    required = {
        'HF_TOKEN':        HF_TOKEN,
        'HF_SOURCE_REPO':  HF_SOURCE_REPO,
        'HF_MODELS_REPO':  HF_MODELS_REPO,
        'HF_RESULTS_REPO': HF_RESULTS_REPO,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(
            f"[config] Missing required environment variables: {missing}\n"
            f"Set these as GitHub Secrets / Kaggle Secrets before running."
        )

    if GEMINI_API_KEY is None:
        print("[config] WARNING: GEMINI_API_KEY not set — "
              "ERL reflection will fall back to rule_based")

    print(f"[config] {PROJECT_NAME} v{VERSION} ✓")
    print(f"[config] Source:      {HF_SOURCE_REPO}")
    print(f"[config] Models:      {HF_MODELS_REPO}")
    print(f"[config] Results:     {HF_RESULTS_REPO}")
    print(f"[config] Assets:      {ALL_ASSETS}")
    print(f"[config] DDPG state:  {DDPG_STATE_DIM} dims")
    print(f"[config] HMM states:  {HMM_N_STATES}")
    print(f"[config] Live start:  {LIVE_START}")


if __name__ == "__main__":
    validate()

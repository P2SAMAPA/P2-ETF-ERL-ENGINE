# REALM — P2 ETF ERL Engine

**Regime-Aware Experiential Asset Learning Machine**

A fully autonomous fixed-income and alternatives portfolio engine that detects macro regimes, trains reinforcement learning policies, and generates daily allocation signals — all running on free infrastructure.

---

## What It Does

REALM observes macro conditions daily, identifies the current market regime, and allocates across 7 assets using an ensemble of trained policies. It learns from its own mistakes through a reflection loop and stores actionable rules in a persistent rulebook.

**Assets:** TLT · LQD · HYG · VNQ · GLD · SLV · CASH  
**Benchmark:** AGG  
**Live since:** 2025-01-01  
**Training data:** 2008–2024  

---

## Architecture

```
Macro Data (TNX, DXY, VIX, Spreads)
        │
        ▼
┌─────────────────┐
│  HMM Regime     │  8 states: Low Vol Expansion → Acute Crisis
│  Detector       │
└────────┬────────┘
         │ regime probabilities
         ▼
┌─────────────────┐
│  TFT Encoder    │  64-dim temporal embedding from price features
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  DDPG Ensemble                      │
│  Policy A — Crisis regimes [4,5,6]  │
│  Policy B — Expansion regimes [0,1,7]│
│  Policy C — All regimes             │
└────────┬────────────────────────────┘
         │ HMM-gated blend
         ▼
┌─────────────────┐
│  ERL Loop       │  Reflect → Improve → Store rules (50bp gate)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kelly Sizing   │  fraction = 0.25 × regime × agreement × sharpe
└────────┬────────┘
         │
         ▼
     Daily Signal → HuggingFace → Streamlit Dashboard
```

---

## Infrastructure

| Component | Service | Cost |
|-----------|---------|------|
| Model training | GitHub Actions (CPU) | Free |
| Daily signal | GitHub Actions (CPU) | Free |
| Model weights | HuggingFace Datasets | Free |
| Signal history | HuggingFace Datasets | Free |
| Dashboard | Streamlit Community Cloud | Free |

**GitHub Actions schedule:**
- `daily_training.yml` — 01:00 UTC daily — trains all models, pushes to HF
- `daily_signal.yml` — 22:00 UTC Mon–Fri — generates allocation signal

---

## Repository Structure

```
P2-ETF-ERL-ENGINE/
│
├── config.py           # All hyperparameters and constants
├── loader.py           # Data loading from HuggingFace source repo
├── features.py         # Feature engineering and scaling
│
├── hmm_train.py        # 8-state Gaussian HMM regime detector
├── tft_train.py        # Temporal Fusion Transformer encoder
├── environment.py      # Portfolio simulation environment
├── ddpg_train.py       # DDPG actor-critic (3 policies: A, B, C)
├── erl_train.py        # Experiential Reflection Loop
├── reflect.py          # Gemini + rule-based reflector
├── memory.py           # Rulebook and audit trail
├── internalize.py      # SFT distillation from ERL demos
│
├── ensemble.py         # HMM-gated policy blending
├── kelly.py            # Kelly position sizing
├── predict.py          # Daily signal generation
├── score.py            # Signal scoring vs benchmark
│
├── streamlit_app.py    # Live dashboard
├── requirements.txt    # Python dependencies
│
└── .github/
    └── workflows/
        ├── daily_training.yml   # 01:00 UTC — full model training
        └── daily_signal.yml     # 22:00 UTC — signal generation
```

---

## Kelly Sizing Formula

```
fraction = BASE × regime_scalar × agreement_scalar × sharpe_scalar

BASE              = 0.25
regime_scalar     = 1 - entropy / ln(8)        # certainty of regime
agreement_scalar  = 0.5 + 0.5 × cosine_sim     # policy consensus
sharpe_scalar     = clip(sharpe / 2, 0.1, 1.0) # recent performance
```

---

## ERL Reflection Loop

Each training episode:
1. Run policy on a random 252-day window
2. If excess return vs AGG < 0 → trigger reflection
3. Gemini (or rule-based fallback) generates an adjustment
4. Run second attempt with adjusted policy
5. If improvement ≥ 50bp → store rule in rulebook
6. Rulebook rules are injected into future signals via `predict.py`

---

## HuggingFace Repos

| Repo | Contents |
|------|----------|
| `P2SAMAPA/p2-etf-deepwave-dl` | Source price + macro data (read-only) |
| `P2SAMAPA/p2-etf-erl-models` | Model weights (HMM, TFT, DDPG policies) |
| `P2SAMAPA/p2-etf-erl-results` | Signals, rulebook, regime history, performance |

---

## Setup

### 1. GitHub Secrets

Add these secrets to your repository (Settings → Secrets → Actions):

```
HF_TOKEN          HuggingFace API token
HF_SOURCE_REPO    P2SAMAPA/p2-etf-deepwave-dl
HF_MODELS_REPO    P2SAMAPA/p2-etf-erl-models
HF_RESULTS_REPO   P2SAMAPA/p2-etf-erl-results
GEMINI_API_KEY    Google AI Studio API key
```

### 2. First Training Run

Trigger manually from GitHub Actions → Daily Training → Run workflow.  
Runtime: ~2.5 hours on CPU.

### 3. Streamlit Dashboard

Deploy `streamlit_app.py` to [Streamlit Community Cloud](https://share.streamlit.io).  
Add the same 4 HF secrets in the app settings (exclude `GEMINI_API_KEY`).

---

## Regime Map

| ID | Name | Policy |
|----|------|--------|
| 0 | Low Vol Expansion | B |
| 1 | Mid Cycle Growth | B |
| 2 | Late Cycle | C |
| 3 | Curve Flattening | C |
| 4 | Credit Stress | A |
| 5 | Risk Off | A |
| 6 | Acute Crisis | A |
| 7 | Recovery | B |

---

## Disclaimer

This project is for research and educational purposes only. It is not financial advice. Past performance of backtested signals does not guarantee future results.

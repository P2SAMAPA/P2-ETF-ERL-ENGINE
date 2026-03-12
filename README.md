# REALM — P2 ETF ERL Engine

**Regime-Aware Experiential Asset Learning Machine**

A fully autonomous fixed-income and alternatives signal engine that detects macro regimes, trains reinforcement learning policies and a supervised ETF classifier, and generates a single daily ETF pick — all running on free infrastructure.

**Assets:** TLT · LQD · HYG · VNQ · GLD · SLV · CASH  
**Benchmark:** AGG  
**Live since:** 2025-01-01  
**Training data:** 2008–2024  

---

## What It Does

Every trading day REALM produces one output: **a single ETF ticker** (or CASH) for the next trading session, with a conviction score and plain-English rationale.

The pick comes from a supervised Transformer classifier trained on 4000+ days of labelled data, backed by an ensemble of DDPG reinforcement learning policies for regime-gated weighting. A Gemini-powered reflection loop extracts reusable rules from each episode and stores them in a persistent rulebook.

---

## Architecture

```
Market Data (HF Dataset)
        │
        ▼
┌───────────────────┐
│  HMM Regime       │  8-state Gaussian HMM on macro features
│  Detector         │  TNX · DXY · CORP_SPREAD · HY_SPREAD
│                   │  VIX · T10Y2Y · TBILL_3M
└────────┬──────────┘
         │  regime_id + 8-dim probability vector
         ▼
┌───────────────────┐
│  TFT Encoder      │  Temporal Fusion Transformer
│                   │  Price + macro → 64-dim daily embedding
└────────┬──────────┘
         │  embedding + HMM probs + macro + 5d return lookback
         ▼
┌─────────────────────────────────────────────┐
│  ETF Classifier (primary)                    │
│  Transformer encoder → 7-class softmax       │
│  Label: argmax(next-day return) or CASH      │
│  Output: pick + conviction (probability)     │
└─────────────────────────────────────────────┘
         │  (fallback if classifier unavailable)
         ▼
┌─────────────────────────────────────────────┐
│  DDPG Ensemble (fallback)                    │
│  Policy A: crisis regimes  [3, 4, 6]         │
│  Policy B: expansion       [0, 5, 7]         │
│  Policy C: all regimes                       │
│  → regime-gated weight vector → argmax pick  │
└─────────────────────────────────────────────┘
         │
         ▼
┌───────────────────┐
│  ERL Reflection   │  30 episodes per training run
│  Loop             │  Gemini / rule-based reflection
│                   │  Rulebook: 20 rules, regime-tagged
└────────┬──────────┘
         │
         ▼
    Daily Signal
    ┌─────────────────────────────┐
    │  PICK:        GLD           │
    │  CONVICTION:  38%           │
    │  REGIME:      Late Cycle    │
    │  SOURCE:      AI Classifier │
    └─────────────────────────────┘
```

---

## Regime Map

| ID | Name | Description | Policy |
|----|------|-------------|--------|
| 0 | Mid Cycle Growth | Moderate yields, low vol | B (Expansion) |
| 1 | Late Cycle | Flattening curve, rising rates | C (Full) |
| 2 | Curve Flattening | Inverted / flat yield curve | C (Full) |
| 3 | Credit Stress | Wide HY spreads | A (Crisis) |
| 4 | Risk Off | Elevated VIX, wide spreads | A (Crisis) |
| 5 | Low Vol Expansion | Low VIX, tight spreads | B (Expansion) |
| 6 | Acute Crisis | Extreme VIX + HY spread | A (Crisis) → CASH override |
| 7 | Recovery | Post-crisis normalisation | B (Expansion) |

---

## File Structure

```
P2-ETF-ERL-ENGINE/
├── config.py               — all hyperparameters and paths
├── loader.py               — HuggingFace data loader
├── features.py             — TFT features, HMM features, scaler
│
├── hmm_train.py            — 8-state Gaussian HMM training
├── tft_train.py            — Temporal Fusion Transformer training
├── ddpg_train.py           — DDPG ensemble training (3 policies)
├── erl_train.py            — ERL reflection loop + rulebook
├── classifier_train.py     — Supervised ETF classifier
│
├── environment.py          — Portfolio RL environment
├── ensemble.py             — Regime-gated ensemble policy
├── memory.py               — Rulebook storage (HuggingFace)
├── reflect.py              — Gemini / rule-based reflection
├── internalize.py          — Rule application to DDPG actions
├── score.py                — Daily pick scoring vs AGG
├── predict.py              — Daily signal generation
│
├── streamlit_app.py        — Live dashboard
├── requirements.txt
│
└── .github/workflows/
    ├── daily_train.yml     — 01:00 UTC daily (HMM→TFT→DDPG→ERL→CLF)
    └── daily_signal.yml    — 22:00 UTC Mon–Fri (score→predict)
```

---

## HuggingFace Repositories

| Repo | Purpose |
|------|---------|
| `P2SAMAPA/p2-etf-deepwave-dl` | Source price + macro data (read-only) |
| `P2SAMAPA/p2-etf-erl-models` | Trained model weights |
| `P2SAMAPA/p2-etf-erl-results` | Daily signals, regime history, rulebook |

### Models (`p2-etf-erl-models`)

```
models/
├── regime_detector.pkl      — HMM + scaler wrapped in RegimeDetector
├── tft_model.pt             — TFT encoder weights
├── tft_scaler.pkl           — TFT feature scaler
├── policy_A.pt              — DDPG crisis policy
├── policy_B.pt              — DDPG expansion policy
├── policy_C.pt              — DDPG full-universe policy
├── etf_classifier.pt        — Transformer ETF classifier weights
└── classifier_meta.json     — accuracy, config, per-class breakdown

data/
└── feature_cache.parquet    — TFT embeddings (4000+ days × 64)
```

### Results (`p2-etf-erl-results`)

```
results/
├── latest_signal.json       — today's pick
├── signal_history.json      — all scored signals
├── regime_history.csv       — daily regime assignments
├── rulebook.json            — 20 stored ERL rules
└── audit_trail.json         — ERL episode log
```

---

## Daily Schedule

| Time (UTC) | Workflow | Steps | Duration |
|------------|----------|-------|----------|
| 01:00 daily | `daily_train.yml` | HMM → TFT → DDPG → ERL → Classifier | ~3 hrs |
| 22:00 Mon–Fri | `daily_signal.yml` | score → predict | ~2 min |

---

## Setup

### 1. Secrets — GitHub + Streamlit

```
HF_TOKEN          — HuggingFace write token
HF_SOURCE_REPO    — P2SAMAPA/p2-etf-deepwave-dl
HF_MODELS_REPO    — P2SAMAPA/p2-etf-erl-models
HF_RESULTS_REPO   — P2SAMAPA/p2-etf-erl-results
GEMINI_API_KEY    — Google AI Studio key (free tier)
```

### 2. First run

Trigger `daily_train.yml` → Run workflow manually in GitHub Actions (~3 hrs).  
Then trigger `daily_signal.yml` → Run workflow.

### 3. Dashboard

Connect `streamlit_app.py` to Streamlit Community Cloud with the same 5 secrets.

---

## Signal Output

`results/latest_signal.json`:

```json
{
  "date": "2026-03-12",
  "pick": "GLD",
  "conviction": 0.38,
  "rationale": "GLD selected by CLASSIFIER (conviction=38%) | 3 active rule(s)",
  "pick_source": "CLASSIFIER",
  "regime": 5,
  "regime_name": "Low Vol Expansion",
  "crisis_prob": 0.0,
  "classifier_probs": {
    "TLT": 0.12, "LQD": 0.09, "HYG": 0.08,
    "VNQ": 0.11, "GLD": 0.38, "SLV": 0.15, "CASH": 0.07
  }
}
```

---

## Methodology

**Classifier vs portfolio weights**  
DDPG outputs a continuous weight vector optimised for portfolio return. Near-uniform weights mean there is no high-conviction pick. The ETF classifier is trained directly on "which asset has the highest return tomorrow?" using cross-entropy loss — its softmax output is a calibrated probability, making conviction scores meaningful.

**CASH**  
Triggered when: (a) regime = Acute Crisis (hard override), or (b) classifier assigns highest probability to CASH (trained on days where all ETFs had negative next-day returns or VIX > 25).

**Regime stability**  
HMM retrained daily with 5 random seeds, best log-likelihood wins. Regime names are assigned by macro characteristics (VIX, HY spread, yield curve slope) so they remain consistent across retraining runs regardless of state ID order.

**ERL + Gemini**  
30 episodes post-DDPG. Improvements above 0.5% vs baseline are formalised as rules via Gemini (gemini-2.0-flash) or a rule-based fallback if API quota is exhausted. Rules are regime-tagged and applied as weight nudges in both DDPG ensemble and classifier inference.

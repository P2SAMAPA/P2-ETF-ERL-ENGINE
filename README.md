# REALM — P2-ETF-ERL-ENGINE

**Regime-Aware Experiential Asset Learning Machine**

A production portfolio engine combining HMM regime detection, Temporal Fusion Transformer state encoding, DDPG ensemble policies, and an Experience-Reflection-Learning loop with Gemini-powered reflection.

---

## Architecture

```
Market Data (TLT LQD HYG VNQ GLD SLV + AGG)
        │
        ├─► HMM (8 regimes)  ──► regime label + 8-dim probs
        │
        ├─► TFT Encoder      ──► 64-dim regime embedding
        │
        └─► Features (41-dim) ─► DDPG state (80-dim)
                                       │
                          ┌────────────┼────────────┐
                       Policy A     Policy B     Policy C
                      (crisis)    (expansion)    (full)
                          └────────────┼────────────┘
                                  Ensemble Gate
                                  (HMM probs)
                                       │
                              ERL Reflection Loop
                                (Gemini Flash)
                                       │
                                 Kelly Sizing
                                       │
                               Daily Signal JSON
```

### Layer Summary

| Layer | Component | Purpose | Runtime |
|-------|-----------|---------|---------|
| 1 | HMM (8 states) | Market regime detection | CPU, <1s |
| 2 | TFT Encoder | 64-dim temporal embedding | GPU, ~3h train |
| 3 | DDPG × 3 | Regime-specialist policies | GPU, ~4h train |
| 4 | ERL Loop | Reflection + improvement | GPU, ~6h train |
| 5 | Ensemble Gate | HMM-weighted policy blend | CPU, <1s |
| 6 | Kelly Sizing | Dynamic position sizing | CPU, <1s |

---

## Asset Universe

| Ticker | Asset | Role |
|--------|-------|------|
| TLT | iShares 20+ Year Treasury | Duration / safe haven |
| LQD | iShares IG Corporate Bond | Credit quality |
| HYG | iShares HY Corporate Bond | Risk appetite proxy |
| VNQ | Vanguard Real Estate ETF | Inflation / real assets |
| GLD | SPDR Gold Trust | Crisis hedge |
| SLV | iShares Silver Trust | Industrial / risk-on |
| CASH | — | Defensive residual |

**Benchmark:** AGG (iShares Core US Aggregate Bond)

---

## File Structure

```
P2-ETF-ERL-ENGINE/
├── config.py              Single source of truth — all hyperparameters
├── loader.py              HF price data loader + train/live splitter
├── features.py            Price + macro feature engineering
├── hmm_train.py           HMM regime detector training + inference
├── tft_train.py           TFT encoder training + embedding cache
├── ddpg_train.py          DDPG ensemble training (Policies A, B, C)
├── erl_train.py           ERL training loop (reflect + improve)
├── internalize.py         SFT distillation of reflections → policy
├── environment.py         Portfolio simulation environment
├── reflect.py             Gemini reflection generator
├── memory.py              Rulebook + audit trail
├── ensemble.py            Regime-gated ensemble + Kelly sizing
├── kelly.py               Fractional Kelly position sizing
├── predict.py             Daily signal generation (GitHub Actions)
├── score.py               Scores yesterday's signal vs actual prices
├── streamlit_app.py       Live dashboard
├── requirements.txt
├── requirements_kaggle.txt
├── kaggle_monday.ipynb    Full training pipeline
├── kaggle_thursday.ipynb  ERL-only incremental update
└── .github/workflows/
    └── daily_signal.yml   Daily signal automation
```

---

## HuggingFace Repos

| Repo | Purpose |
|------|---------|
| `P2SAMAPA/p2-etf-deepwave-dl` | Source prices (read-only) |
| `P2SAMAPA/p2-etf-erl-models` | Model weights + embedding cache |
| `P2SAMAPA/p2-etf-erl-results` | Signals, history, rulebook |

---

## Setup

### 1. GitHub Secrets

Add to `P2-ETF-ERL-ENGINE` → Settings → Secrets:

```
HF_TOKEN
HF_SOURCE_REPO   = P2SAMAPA/p2-etf-deepwave-dl
HF_MODELS_REPO   = P2SAMAPA/p2-etf-erl-models
HF_RESULTS_REPO  = P2SAMAPA/p2-etf-erl-results
GEMINI_API_KEY
```

### 2. Kaggle Secrets

Same 5 secrets added to both notebooks (`kaggle_monday`, `kaggle_thursday`).

### 3. First Run Order

```bash
# On Kaggle GPU (Monday notebook):
# Cells run in order: HMM → TFT → DDPG → ERL
# ~13 hours total

# Afterwards, daily GitHub Actions runs automatically:
# score.py → predict.py  (~2 min, CPU)
```

### 4. Dashboard

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Or deploy to Streamlit Community Cloud by connecting to this repo.

---

## Training Schedule

| Day | Notebook | What runs | GPU hrs |
|-----|----------|-----------|---------|
| Monday 02:00 UTC | `kaggle_monday` | HMM + TFT + DDPG + ERL | ~13h |
| Thursday 02:00 UTC | `kaggle_thursday` | ERL only (incremental) | ~6h |
| Mon–Fri 22:00 UTC | GitHub Actions | score + predict | ~2min |

---

## Kelly Fraction Formula

```
fraction = BASE(0.25) × regime_scalar × agreement_scalar × sharpe_scalar

regime_scalar    = 1 - (transition_entropy / ln(8))
agreement_scalar = 0.5 + 0.5 × mean_cosine_sim(A, B, C)
sharpe_scalar    = clip(rolling_sharpe / 2, 0.1, 1.0)

Final: clip(fraction, 0.05, 0.40)
Residual → CASH (capped at 30%)
```

---

## ERL Loop

```
Episode (252 trading days):
  1. First attempt  → policy forward pass
  2. r < 0?         → call Gemini Flash for reflection
  3. Second attempt → apply weight adjustment from reflection
  4. Δexcess ≥ 50bp → store rule in rolling rulebook (max 20)
  5. Every N ep.    → SFT distillation: successful actions → actor weights
```

Reflection provider: Gemini 1.5 Flash (free tier: 1500 req/day).
Falls back to `rule_based` if API unavailable.

---

## License

Private research project. Not financial advice.

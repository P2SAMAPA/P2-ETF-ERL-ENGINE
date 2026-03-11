# erl_train.py — ERL Training Loop for P2-ETF-ERL-ENGINE
# Orchestrates the full Experience-Reflection-Learning cycle:
#
#   For each episode:
#     1. First attempt  → run policy → collect reward + regime feedback
#     2. Reflect        → if r < threshold: call Gemini/rule-based reflector
#     3. Second attempt → run policy conditioned on reflection adjustment
#     4. Store          → if improvement >= 50bps: add to rulebook
#     5. Internalize    → every N episodes: SFT distillation (see internalize.py)
#
# Run on Kaggle GPU after hmm_train.py + tft_train.py + ddpg_train.py:
#   python erl_train.py

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from loader import load_all
from features import compute_hmm_features
from hmm_train import RegimeDetector
from environment import PortfolioEnv, ERLEpisode, compute_benchmark_return
from ddpg_train import DDPGAgent, load_training_data
from reflect import Reflector
from memory import Rulebook, AuditTrail

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[ERL] Using device: {DEVICE}")


# ── ERL Episode Runner ─────────────────────────────────────────────────────────

def run_episode(
    agent:       DDPGAgent,
    env:         PortfolioEnv,
    adjustment:  dict = None,
    training:    bool = True,
) -> tuple[float, float, list, list, list]:
    """
    Run one full episode with optional reflection-based weight adjustment.

    Parameters
    ----------
    agent      : DDPGAgent — policy to run
    env        : PortfolioEnv — simulation environment
    adjustment : dict or None — weight adjustments from reflector
    training   : bool — whether to add OU noise

    Returns
    -------
    (total_return, bench_return, actions, rewards, states)
    """
    state   = env.reset()
    done    = False
    actions = []
    rewards = []
    states  = []

    # Parse adjustment vector if provided
    adj_vector = None
    if adjustment and 'weight_adjustments' in adjustment:
        adj_dict  = adjustment['weight_adjustments']
        adj_vector = np.array(
            [adj_dict.get(a, 0.0) for a in cfg.ALL_ASSETS],
            dtype=np.float32,
        )
        adj_scale = float(adjustment.get('conviction', 0.5))
        adj_vector = adj_vector * adj_scale

    while not done:
        # Select base action from policy
        action = agent.select_action(state, training=training)

        # Apply reflection adjustment (additive, then re-softmax)
        if adj_vector is not None:
            action = action + adj_vector
            action = np.exp(action) / (np.exp(action).sum() + 1e-8)

        next_state, reward, done, info = env.step(action)

        if training:
            agent.buffer.push(state, action, reward, next_state, done)
            agent.update()

        actions.append(action.copy())
        rewards.append(reward)
        states.append(state.copy())
        state = next_state

    # Compute benchmark return over episode period
    bench_return = _compute_episode_bench(env)

    return (
        env.total_return(),
        bench_return,
        actions,
        rewards,
        states,
    )


def _compute_episode_bench(env: PortfolioEnv) -> float:
    """Compute benchmark total return for the episode period."""
    bench_returns = env.bench_returns[: env.t]
    if len(bench_returns) == 0:
        return 0.0
    return float((1 + bench_returns).prod() - 1)


# ── Attention Summary ──────────────────────────────────────────────────────────

def build_attention_summary(
    tft_attention: np.ndarray,
    dates:         pd.DatetimeIndex,
    top_k:         int = 3,
) -> str:
    """
    Summarise TFT attention weights for the reflection prompt.
    Identifies which historical periods the model focused on.
    """
    if tft_attention is None or len(tft_attention) == 0:
        return "Attention data not available."

    # Average over heads if multi-head
    if tft_attention.ndim == 3:
        attn = tft_attention.mean(axis=0)
    else:
        attn = tft_attention

    # Attention is (T, T) — rows=query, cols=key
    # Average over query dimension to get key importance
    key_importance = attn.mean(axis=0)

    # Map back to dates
    n = min(len(key_importance), len(dates))
    importances = key_importance[-n:]
    date_labels = dates[-n:]

    # Find top-k most attended periods
    top_idx = np.argsort(importances)[-top_k:][::-1]
    lines   = []
    for idx in top_idx:
        if idx < len(date_labels):
            d   = date_labels[idx]
            pct = float(importances[idx]) / (importances.sum() + 1e-8)
            lines.append(f"{d.strftime('%Y-%m')} ({pct:.0%})")

    return "Most attended periods: " + ", ".join(lines)


# ── Episode Data Builder ───────────────────────────────────────────────────────

def build_episode_data(
    ep:            ERLEpisode,
    env:           PortfolioEnv,
    rulebook:      Rulebook,
    detector_output: dict,
) -> dict:
    """
    Assemble the episode_data dict required by Reflector.generate_reflection().
    """
    # Mean weights over episode
    if ep.first_actions:
        mean_weights = np.mean(ep.first_actions, axis=0)
        allocation   = {a: float(mean_weights[i])
                        for i, a in enumerate(cfg.ALL_ASSETS)}
    else:
        allocation = {a: 1/cfg.N_ASSETS for a in cfg.ALL_ASSETS}

    # Worst single day
    daily_rets = np.array(env.daily_returns) if env.daily_returns else np.array([0.0])
    worst_day  = float(daily_rets.min())

    # Regime transition risk label
    entropy     = detector_output.get('transition_entropy', 0.0)
    crisis_prob = detector_output.get('crisis_prob', 0.0)
    if crisis_prob > 0.4 or entropy > 1.8:
        transition_risk = (
            f"HIGH — {crisis_prob:.0%} probability of transition to crisis"
        )
    elif crisis_prob > 0.2:
        transition_risk = f"MODERATE — {crisis_prob:.0%} crisis probability"
    else:
        transition_risk = f"LOW — {crisis_prob:.0%} crisis probability"

    # Existing rules for this regime
    existing_rules = rulebook.get_rules_for_regime(ep.regime, max_rules=3)

    return {
        'regime_id':         ep.regime,
        'regime_name':       ep.regime_name,
        'transition_risk':   transition_risk,
        'asset_allocation':  allocation,
        'portfolio_return':  float(ep.first_return),
        'bench_return':      float(
                                _compute_episode_bench(env)
                             ),
        'excess_return':     float(ep.first_excess),
        'n_days':            env.t,
        'worst_day':         worst_day,
        'attention_summary': build_attention_summary(
                                 ep.tft_attention,
                                 env.dates[:env.t],
                             ),
        'existing_rules':    existing_rules,
    }


# ── Select Policy for Regime ───────────────────────────────────────────────────

def select_policy(
    regime_id: int,
    agents:    dict,
    hmm_probs: np.ndarray,
) -> DDPGAgent:
    """
    Select which policy to use for the current regime.
    Falls back to Policy C if specialist not available.
    """
    crisis_prob    = float(sum(
        hmm_probs[k] for k in cfg.POLICY_A_REGIMES
        if k < len(hmm_probs)
    ))
    expansion_prob = float(sum(
        hmm_probs[k] for k in cfg.POLICY_B_REGIMES
        if k < len(hmm_probs)
    ))

    if crisis_prob > cfg.ENSEMBLE_CRISIS_THRESHOLD and 'A' in agents:
        return agents['A']
    elif expansion_prob > 0.4 and 'B' in agents:
        return agents['B']
    return agents.get('C', list(agents.values())[0])


# ── Main ERL Loop ──────────────────────────────────────────────────────────────

def run_erl_loop(
    agents:      dict,          # {'A': DDPGAgent, 'B': ..., 'C': ...}
    data:        dict,
    rulebook:    Rulebook,
    detector:    RegimeDetector,
    reflector:   Reflector,
    audit:       AuditTrail,
    n_episodes:  int = None,
) -> dict:
    """
    Run the full ERL training loop.

    Each episode covers a random 252-day window from the training period.
    Episodes are stratified by regime to ensure all regimes are covered.
    """
    if n_episodes is None:
        n_episodes = cfg.DDPG_MAX_EPOCHS

    stats = {
        'episodes':       0,
        'reflected':      0,
        'improved':       0,
        'stored':         0,
        'mean_excess':    [],
        'mean_improve':   [],
    }

    # Build a pool of episode start dates from training period
    train_dates = data['prices'].index[
        (data['prices'].index >= cfg.DATA_START) &
        (data['prices'].index <= cfg.TRAIN_END)
    ]
    ep_len   = 252   # one year per episode

    # Compute HMM labels for all training dates
    hmm_feats  = compute_hmm_features(data['prices'], data['benchmark'])
    X_all      = detector.scaler.transform(
                     hmm_feats.reindex(train_dates).dropna()
                 ).values
    all_labels = detector.model.predict(X_all)
    label_series = pd.Series(
        all_labels,
        index = hmm_feats.reindex(train_dates).dropna().index,
        name  = 'regime',
    )

    print(f"\n[ERL] Starting loop: {n_episodes} episodes")

    for ep_num in range(1, n_episodes + 1):

        # ── Sample episode window ──────────────────────────────────────────
        max_start = len(train_dates) - ep_len - 1
        if max_start <= 0:
            break
        start_idx  = np.random.randint(0, max_start)
        ep_dates   = train_dates[start_idx : start_idx + ep_len]
        ep_start   = ep_dates[0]
        ep_end     = ep_dates[-1]

        # Dominant regime in this window
        window_labels = label_series.reindex(ep_dates).dropna()
        if len(window_labels) == 0:
            continue
        regime_id   = int(window_labels.mode()[0])
        regime_name = cfg.REGIME_NAMES.get(regime_id, f'Regime {regime_id}')

        # HMM probs at start of window
        hmm_p_series = data['hmm_probs'].reindex(ep_dates).dropna()
        if len(hmm_p_series) == 0:
            continue
        hmm_p_latest = hmm_p_series.iloc[-1].values

        # Select agent
        agent = select_policy(regime_id, agents, hmm_p_latest)

        # Build environment for this window
        env = PortfolioEnv(
            embeddings    = data['embeddings'].reindex(ep_dates),
            hmm_probs     = data['hmm_probs'].reindex(ep_dates),
            asset_returns = data['returns'].reindex(ep_dates),
            bench_returns = data['bench_returns'].reindex(ep_dates),
        )

        # ── First attempt ──────────────────────────────────────────────────
        first_return, bench_return, actions, rewards, states = run_episode(
            agent, env, adjustment=None, training=True
        )
        first_excess = first_return - bench_return

        ep = ERLEpisode(
            episode_id  = ep_num,
            regime      = regime_id,
            regime_name = regime_name,
        )
        ep.record_first_attempt(
            actions       = actions,
            rewards       = rewards,
            states        = states,
            total_return  = first_return,
            excess_return = first_excess,
        )

        stats['episodes'] += 1
        stats['mean_excess'].append(first_excess)

        # ── Reflect? ───────────────────────────────────────────────────────
        needs_reflection = (first_excess < cfg.ERL_REWARD_THRESHOLD)

        if needs_reflection:
            stats['reflected'] += 1

            # Get detector output for reflection context
            det_out = detector.predict(
                hmm_feats.reindex(ep_dates).dropna().tail(
                    cfg.TFT_CONTEXT_LENGTH
                )
            )

            # Build episode data for reflection prompt
            ep_data     = build_episode_data(
                ep, env, rulebook, det_out
            )
            reflection  = reflector.generate_reflection(ep_data)
            ep.record_reflection(
                reflection.get('action', reflection.get('rationale', ''))
            )

            # State data for second attempt adjustment
            state_data = {
                'regime_name':   regime_name,
                'crisis_prob':   det_out.get('crisis_prob', 0.0),
                'entropy':       det_out.get('transition_entropy', 0.0),
                'sharpe':        env.sharpe_ratio(),
                'current_weights': {
                    a: float(
                        np.mean([ac[i] for ac in actions], axis=0)
                    ) if actions else 1/cfg.N_ASSETS
                    for i, a in enumerate(cfg.ALL_ASSETS)
                },
            }
            active_rules = rulebook.get_rules_for_regime(regime_id)
            adjustment   = reflector.generate_second_attempt_adjustment(
                state_data, active_rules + [reflection]
            )

            # ── Second attempt ─────────────────────────────────────────────
            for _ in range(cfg.ERL_SECOND_ATTEMPT_EPISODES):
                s_return, s_bench, s_actions, s_rewards, _ = run_episode(
                    agent, env, adjustment=adjustment, training=True
                )
                s_excess = s_return - s_bench

            ep.record_second_attempt(
                actions       = s_actions,
                rewards       = s_rewards,
                total_return  = s_return,
                excess_return = s_excess,
            )

            improvement = ep.improvement
            stats['mean_improve'].append(improvement)

            if ep.improved:
                stats['improved'] += 1

            # ── Store rule? ────────────────────────────────────────────────
            stored = rulebook.add(
                rule       = reflection,
                improvement = improvement,
                episode_id  = ep_num,
            )
            if stored:
                stats['stored'] += 1

            audit.record(
                episode_id   = ep_num,
                regime_id    = regime_id,
                regime_name  = regime_name,
                first_excess = first_excess,
                second_excess = s_excess,
                reflection   = reflection,
                stored       = stored,
            )

        else:
            audit.record(
                episode_id   = ep_num,
                regime_id    = regime_id,
                regime_name  = regime_name,
                first_excess = first_excess,
                second_excess = None,
                reflection   = None,
                stored       = False,
            )

        # ── Progress log ───────────────────────────────────────────────────
        if ep_num % 10 == 0:
            mean_excess   = np.mean(stats['mean_excess'][-10:])
            reflect_rate  = stats['reflected'] / stats['episodes']
            improve_rate  = (
                stats['improved'] / max(stats['reflected'], 1)
            )
            print(
                f"[ERL] Ep {ep_num:4d}/{n_episodes} | "
                f"Regime: {regime_name:20s} | "
                f"Excess: {first_excess:+.2%} | "
                f"Mean(10): {mean_excess:+.2%} | "
                f"Reflect: {reflect_rate:.0%} | "
                f"Improve: {improve_rate:.0%} | "
                f"Rules: {len(rulebook.rules)}"
            )

    return stats


# ── HuggingFace Push ───────────────────────────────────────────────────────────

def push_to_hf(local_path: str, repo_id: str, repo_path: str):
    api = HfApi(token=cfg.HF_TOKEN)
    api.upload_file(
        path_or_fileobj = local_path,
        path_in_repo    = repo_path,
        repo_id         = repo_id,
        repo_type       = "dataset",
    )
    print(f"[HF] Pushed {repo_path} → {repo_id}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg.validate()
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[ERL] Loading data...")
    data = load_training_data()

    # ── 2. Load HMM detector ──────────────────────────────────────────────────
    print("\n[ERL] Loading regime detector...")
    det_path = hf_hub_download(
        repo_id      = cfg.HF_MODELS_REPO,
        filename     = "models/regime_detector.pkl",
        repo_type    = "dataset",
        token        = cfg.HF_TOKEN,
        force_download = True,
    )
    detector = RegimeDetector.load(det_path)

    # Attach hmm_probs to data dict for episode building
    hmm_feats  = compute_hmm_features(data['prices'], data['benchmark'])
    X_all      = detector.scaler.transform(hmm_feats).values
    probs_arr  = detector.model.predict_proba(X_all)
    data['hmm_probs'] = pd.DataFrame(
        probs_arr,
        index   = hmm_feats.index,
        columns = list(range(cfg.HMM_N_STATES)),
    )

    # ── 3. Load DDPG agents ───────────────────────────────────────────────────
    print("\n[ERL] Loading DDPG policies...")
    agents = {}
    for name, hf_path in [
        ('A', cfg.POLICY_A_PATH),
        ('B', cfg.POLICY_B_PATH),
        ('C', cfg.POLICY_C_PATH),
    ]:
        try:
            local = hf_hub_download(
                repo_id      = cfg.HF_MODELS_REPO,
                filename     = hf_path,
                repo_type    = "dataset",
                token        = cfg.HF_TOKEN,
                force_download = True,
            )
            agent = DDPGAgent(policy_name=name)
            agent.load(local)
            agents[name] = agent
            print(f"[ERL] Policy {name} loaded ✓")
        except Exception as e:
            print(f"[ERL] Could not load Policy {name}: {e}")

    if not agents:
        raise RuntimeError("[ERL] No policies loaded — run ddpg_train.py first")

    # ── 4. Load or create rulebook ────────────────────────────────────────────
    print("\n[ERL] Loading rulebook...")
    rulebook = Rulebook.load_from_hf()
    rulebook.print_summary()

    # ── 5. Initialise reflector + audit ───────────────────────────────────────
    reflector = Reflector()
    audit     = AuditTrail()

    # ── 6. Run ERL loop ───────────────────────────────────────────────────────
    print(f"\n[ERL] Starting ERL loop ({cfg.DDPG_MAX_EPOCHS} episodes)...")
    start_time = datetime.utcnow()

    stats = run_erl_loop(
        agents     = agents,
        data       = data,
        rulebook   = rulebook,
        detector   = detector,
        reflector  = reflector,
        audit      = audit,
    )

    elapsed = (datetime.utcnow() - start_time).total_seconds() / 60

    # ── 7. Final stats ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ERL Loop Complete ({elapsed:.1f} min)")
    print(f"{'='*60}")
    print(f"  Episodes:         {stats['episodes']}")
    print(f"  Reflected:        {stats['reflected']} "
          f"({stats['reflected']/max(stats['episodes'],1):.0%})")
    print(f"  Improved:         {stats['improved']} "
          f"({stats['improved']/max(stats['reflected'],1):.0%} of reflected)")
    print(f"  Rules stored:     {stats['stored']}")
    print(f"  Rulebook size:    {len(rulebook.rules)}")
    print(f"  Mean excess ret:  {np.mean(stats['mean_excess']):.2%}")
    if stats['mean_improve']:
        print(f"  Mean improvement: {np.mean(stats['mean_improve']):.2%}")

    reflect_stats = reflector.stats()
    print(f"\n  Reflection stats:")
    print(f"    Gemini calls:   {reflect_stats['gemini_calls']}")
    print(f"    Fallback calls: {reflect_stats['fallback_calls']}")
    print(f"    Gemini rate:    {reflect_stats['gemini_rate']:.0%}")

    rulebook.print_summary()

    # ── 8. Save updated policies locally ─────────────────────────────────────
    print("\n[ERL] Saving updated policies...")
    for name, agent in agents.items():
        path = {
            'A': cfg.POLICY_A_PATH,
            'B': cfg.POLICY_B_PATH,
            'C': cfg.POLICY_C_PATH,
        }[name]
        local_path = os.path.join(cfg.LOCAL_TMP, f"policy_{name}.pt")
        agent.save(local_path)
        push_to_hf(local_path, cfg.HF_MODELS_REPO, path)

    # ── 9. Push rulebook + audit to HF ───────────────────────────────────────
    print("\n[ERL] Pushing rulebook + audit trail...")
    rulebook.push_to_hf()
    audit.push_to_hf()

    # ── 10. Save training summary ─────────────────────────────────────────────
    summary = {
        'trained_at':       datetime.utcnow().isoformat(),
        'elapsed_min':      float(elapsed),
        'episodes':         stats['episodes'],
        'reflected':        stats['reflected'],
        'improved':         stats['improved'],
        'rules_stored':     stats['stored'],
        'rulebook_size':    len(rulebook.rules),
        'mean_excess':      float(np.mean(stats['mean_excess'])),
        'reflect_stats':    reflect_stats,
        'audit_summary':    audit.summary(),
    }
    summary_path = os.path.join(cfg.LOCAL_TMP, "erl_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    push_to_hf(summary_path, cfg.HF_MODELS_REPO, "models/erl_summary.json")

    print("\n[ERL] ✅ Complete — all outputs pushed to HuggingFace")


if __name__ == "__main__":
    main()

# ensemble.py — Regime-Gated Ensemble for P2-ETF-ERL-ENGINE
# Combines Policy A (crisis), Policy B (expansion), Policy C (full)
# using HMM regime probabilities as dynamic gates.
#
# Gating logic:
#   gate_A = P(crisis regimes)          — from HMM posterior
#   gate_B = P(expansion regimes)
#   gate_C = 1 - gate_A - gate_B        — residual
#   Weights renormalised to sum to 1
#
# Final action = gate_A * action_A + gate_B * action_B + gate_C * action_C
# Then Kelly-scaled by ensemble agreement + regime entropy + rolling Sharpe
#
# Used by: predict.py

import os
import sys
import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from ddpg_train import DDPGAgent
from kelly import apply_kelly, format_allocation

DEVICE = torch.device('cpu')   # inference always on CPU


# ── Gate Computation ───────────────────────────────────────────────────────────

def compute_gates(hmm_probs: np.ndarray) -> dict:
    """
    Compute policy gate weights from HMM regime probabilities.

    Parameters
    ----------
    hmm_probs : np.ndarray (HMM_N_STATES,) — posterior regime probs

    Returns
    -------
    dict: {'A': float, 'B': float, 'C': float} — sum to 1
    """
    gate_A = float(sum(
        hmm_probs[k]
        for k in cfg.POLICY_A_REGIMES
        if k < len(hmm_probs)
    ))
    gate_B = float(sum(
        hmm_probs[k]
        for k in cfg.POLICY_B_REGIMES
        if k < len(hmm_probs)
    ))
    gate_C = max(0.0, 1.0 - gate_A - gate_B)

    total  = gate_A + gate_B + gate_C + 1e-8
    return {
        'A': gate_A / total,
        'B': gate_B / total,
        'C': gate_C / total,
    }


# ── Ensemble ───────────────────────────────────────────────────────────────────

class EnsemblePolicy:
    """
    Regime-gated ensemble of three DDPG policies.
    Loads policies from HuggingFace on init.
    CPU-only — designed for fast daily inference on GitHub Actions.
    """

    def __init__(self, agents: dict = None):
        """
        Parameters
        ----------
        agents : dict {'A': DDPGAgent, 'B': DDPGAgent, 'C': DDPGAgent}
                 If None, loads from HuggingFace.
        """
        if agents is not None:
            self.agents = agents
        else:
            self.agents = self._load_from_hf()

        self._set_eval()

    def _load_from_hf(self) -> dict:
        """Download all three policy checkpoints from HF_MODELS_REPO."""
        agents  = {}
        paths   = {
            'A': cfg.POLICY_A_PATH,
            'B': cfg.POLICY_B_PATH,
            'C': cfg.POLICY_C_PATH,
        }
        for name, hf_path in paths.items():
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
                # Move to CPU for inference
                agent.actor = agent.actor.cpu()
                agents[name] = agent
                print(f"[ensemble] Policy {name} loaded ✓")
            except Exception as e:
                print(f"[ensemble] Could not load Policy {name}: {e}")

        if 'C' not in agents:
            raise RuntimeError(
                "[ensemble] Policy C (baseline) is required — "
                "run ddpg_train.py first"
            )
        return agents

    def _set_eval(self):
        for agent in self.agents.values():
            agent.actor.eval()

    def get_actions(self, state: np.ndarray) -> dict:
        """
        Get raw action from each available policy.

        Parameters
        ----------
        state : np.ndarray (DDPG_STATE_DIM,)

        Returns
        -------
        dict: {'A': np.ndarray, 'B': np.ndarray, 'C': np.ndarray}
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        actions = {}

        with torch.no_grad():
            for name, agent in self.agents.items():
                action = agent.actor(state_t).squeeze(0).numpy()
                actions[name] = action

        # Fill missing policies with Policy C
        for name in ('A', 'B', 'C'):
            if name not in actions:
                actions[name] = actions['C'].copy()

        return actions

    def forward(
        self,
        state:              np.ndarray,
        hmm_probs:          np.ndarray,
        transition_entropy: float,
        rolling_sharpe:     float,
    ) -> dict:
        """
        Full ensemble forward pass with Kelly sizing.

        Parameters
        ----------
        state              : (DDPG_STATE_DIM,) — current state vector
        hmm_probs          : (HMM_N_STATES,) — regime posterior probs
        transition_entropy : float — from RegimeDetector output
        rolling_sharpe     : float — recent portfolio Sharpe

        Returns
        -------
        dict with:
            final_weights   : np.ndarray (N_ASSETS,) — Kelly-scaled
            raw_weights     : np.ndarray — pre-Kelly ensemble weights
            actions         : dict of per-policy actions
            gates           : dict of regime gates
            kelly_info      : dict of Kelly component scalars
            allocation      : dict — clean formatted allocation
            crisis_prob     : float
        """
        # ── Per-policy actions ─────────────────────────────────────────────
        actions = self.get_actions(state)

        # ── Regime gates ───────────────────────────────────────────────────
        gates = compute_gates(hmm_probs)

        # ── Gated ensemble action ──────────────────────────────────────────
        raw = (
            gates['A'] * actions['A'] +
            gates['B'] * actions['B'] +
            gates['C'] * actions['C']
        )
        # Renormalise via softmax to ensure valid distribution
        raw = np.exp(raw) / (np.exp(raw).sum() + 1e-8)

        # ── Kelly sizing ───────────────────────────────────────────────────
        final_weights, kelly_info = apply_kelly(
            raw_weights        = raw,
            transition_entropy = transition_entropy,
            action_A           = actions['A'],
            action_B           = actions['B'],
            action_C           = actions['C'],
            rolling_sharpe     = rolling_sharpe,
        )

        crisis_prob = float(sum(
            hmm_probs[k] for k in cfg.POLICY_A_REGIMES
            if k < len(hmm_probs)
        ))

        return {
            'final_weights':    final_weights,
            'raw_weights':      raw,
            'actions':          actions,
            'gates':            gates,
            'kelly_info':       kelly_info,
            'allocation':       format_allocation(final_weights),
            'crisis_prob':      crisis_prob,
        }


# ── Smoke Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    np.random.seed(cfg.RANDOM_SEED)

    print("[ensemble] Running smoke test (no HF download)...")

    # Build dummy agents
    agents = {
        name: DDPGAgent(policy_name=name)
        for name in ('A', 'B', 'C')
    }

    ensemble = EnsemblePolicy(agents=agents)

    # Dummy state
    state = np.random.randn(cfg.DDPG_STATE_DIM).astype(np.float32)

    # Test with crisis regime dominant
    hmm_probs_crisis = np.zeros(cfg.HMM_N_STATES)
    hmm_probs_crisis[5] = 0.7   # Risk Off
    hmm_probs_crisis[6] = 0.3   # Acute Crisis

    result = ensemble.forward(
        state              = state,
        hmm_probs          = hmm_probs_crisis,
        transition_entropy = 0.5,
        rolling_sharpe     = 0.8,
    )

    print("\n── Crisis Regime ─────────────────────────────────────────")
    print(f"Gates:       {result['gates']}")
    print(f"Crisis prob: {result['crisis_prob']:.1%}")
    print(f"Kelly frac:  {result['kelly_info']['fraction']:.3f}")
    print(f"Allocation:  {result['allocation']}")
    assert abs(result['final_weights'].sum() - 1.0) < 1e-5

    # Test with expansion regime dominant
    hmm_probs_expand = np.zeros(cfg.HMM_N_STATES)
    hmm_probs_expand[0] = 0.5   # Low Vol Expansion
    hmm_probs_expand[1] = 0.5   # Mid Cycle Growth

    result2 = ensemble.forward(
        state              = state,
        hmm_probs          = hmm_probs_expand,
        transition_entropy = 0.2,
        rolling_sharpe     = 1.8,
    )

    print("\n── Expansion Regime ─────────────────────────────────────")
    print(f"Gates:       {result2['gates']}")
    print(f"Crisis prob: {result2['crisis_prob']:.1%}")
    print(f"Kelly frac:  {result2['kelly_info']['fraction']:.3f}")
    print(f"Allocation:  {result2['allocation']}")
    assert abs(result2['final_weights'].sum() - 1.0) < 1e-5

    # Gate computation test
    uniform = np.ones(cfg.HMM_N_STATES) / cfg.HMM_N_STATES
    gates   = compute_gates(uniform)
    print(f"\n── Uniform Probs Gates ──────────────────────────────────")
    print(f"Gates: {gates}")
    assert abs(sum(gates.values()) - 1.0) < 1e-5

    print("\n✅ Ensemble smoke test passed")

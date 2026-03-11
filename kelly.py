# kelly.py — Fractional Kelly Position Sizing for P2-ETF-ERL-ENGINE
# Scales the ensemble action to a final allocation using Kelly criterion.
#
# Kelly fraction = base × regime_scalar × agreement_scalar × sharpe_scalar
#
# Residual (1 - kelly_fraction) goes to CASH.
# Used by: predict.py, score.py

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def regime_scalar(transition_entropy: float) -> float:
    """
    Scale down Kelly when regime is uncertain (high transition entropy).
    entropy = 0   → scalar = 1.0  (very certain)
    entropy = ln8 → scalar = 0.0  (maximum uncertainty)

    Max entropy for 8 states = ln(8) ≈ 2.079
    """
    max_entropy = np.log(cfg.HMM_N_STATES)
    scalar      = 1.0 - (transition_entropy / (max_entropy + 1e-8))
    return float(np.clip(scalar, 0.1, 1.0))


def agreement_scalar(
    action_A: np.ndarray,
    action_B: np.ndarray,
    action_C: np.ndarray,
) -> float:
    """
    Scale up when the three ensemble policies agree on allocations.
    Agreement = 1 - mean pairwise cosine distance.

    Perfect agreement → scalar = 1.0
    Random disagreement → scalar ≈ 0.5
    """
    def cosine_sim(a, b):
        return float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        )

    sim_AB = cosine_sim(action_A, action_B)
    sim_AC = cosine_sim(action_A, action_C)
    sim_BC = cosine_sim(action_B, action_C)
    mean_sim = (sim_AB + sim_AC + sim_BC) / 3.0

    # Map from [-1, 1] cosine similarity to [0.5, 1.0] scalar
    scalar = 0.5 + 0.5 * mean_sim
    return float(np.clip(scalar, 0.5, 1.0))


def sharpe_scalar(rolling_sharpe: float) -> float:
    """
    Scale Kelly fraction by recent portfolio Sharpe.
    Sharpe >= 2 → scalar = 1.0  (full conviction)
    Sharpe  = 0 → scalar = 0.5  (half conviction)
    Sharpe  < 0 → scalar = 0.1  (near-minimum)

    Formula: clip(sharpe / 2, 0.1, 1.0)
    """
    return float(np.clip(rolling_sharpe / 2.0, 0.1, 1.0))


def kelly_fraction(
    transition_entropy: float,
    action_A:           np.ndarray,
    action_B:           np.ndarray,
    action_C:           np.ndarray,
    rolling_sharpe:     float,
) -> dict:
    """
    Compute the final Kelly fraction and its component scalars.

    Returns
    -------
    dict with:
        fraction        : float — final Kelly fraction [MIN, MAX]
        base            : float — cfg.KELLY_BASE_FRACTION
        regime_scalar   : float
        agreement_scalar: float
        sharpe_scalar   : float
    """
    r_sc = regime_scalar(transition_entropy)
    a_sc = agreement_scalar(action_A, action_B, action_C)
    s_sc = sharpe_scalar(rolling_sharpe)

    fraction = cfg.KELLY_BASE_FRACTION * r_sc * a_sc * s_sc
    fraction = float(np.clip(fraction, cfg.KELLY_MIN_FRACTION,
                             cfg.KELLY_MAX_FRACTION))

    return {
        'fraction':         fraction,
        'base':             cfg.KELLY_BASE_FRACTION,
        'regime_scalar':    r_sc,
        'agreement_scalar': a_sc,
        'sharpe_scalar':    s_sc,
    }


def apply_kelly(
    raw_weights:        np.ndarray,   # (N_ASSETS,) from ensemble
    transition_entropy: float,
    action_A:           np.ndarray,
    action_B:           np.ndarray,
    action_C:           np.ndarray,
    rolling_sharpe:     float,
) -> tuple[np.ndarray, dict]:
    """
    Apply Kelly sizing to the raw ensemble weights.

    The Kelly fraction determines how aggressively to deploy capital.
    The residual (1 - fraction) is added to CASH.

    Parameters
    ----------
    raw_weights : (N_ASSETS,) ensemble softmax weights, sum=1
    ...         : Kelly component inputs

    Returns
    -------
    final_weights : np.ndarray (N_ASSETS,) — Kelly-scaled, sum=1
    kelly_info    : dict — component scalars for audit
    """
    kf        = kelly_fraction(
        transition_entropy, action_A, action_B, action_C, rolling_sharpe
    )
    fraction  = kf['fraction']
    cash_idx  = cfg.ALL_ASSETS.index(cfg.CASH)

    # Scale all weights by Kelly fraction
    scaled = raw_weights.copy() * fraction

    # Add residual to CASH
    residual      = 1.0 - fraction
    scaled[cash_idx] = scaled[cash_idx] + residual

    # Cap CASH at max
    if scaled[cash_idx] > cfg.KELLY_MAX_CASH:
        excess            = scaled[cash_idx] - cfg.KELLY_MAX_CASH
        scaled[cash_idx]  = cfg.KELLY_MAX_CASH
        # Redistribute excess proportionally to non-cash assets
        non_cash    = np.array(
            [1.0 if i != cash_idx else 0.0 for i in range(cfg.N_ASSETS)]
        )
        non_cash_sum = scaled[non_cash > 0].sum()
        if non_cash_sum > 1e-8:
            scaled += non_cash * excess / non_cash_sum

    # Final renormalise
    total = scaled.sum()
    if total > 1e-8:
        final = scaled / total
    else:
        final          = np.zeros(cfg.N_ASSETS)
        final[cash_idx] = 1.0

    return final.astype(np.float32), kf


def format_allocation(weights: np.ndarray, threshold: float = 0.005) -> dict:
    """
    Format weights as a clean allocation dict, filtering near-zero positions.

    Returns
    -------
    dict: {asset: weight} for weights > threshold, sorted desc
    """
    allocation = {
        a: float(weights[i])
        for i, a in enumerate(cfg.ALL_ASSETS)
        if weights[i] > threshold
    }
    return dict(sorted(allocation.items(), key=lambda x: -x[1]))


# ── Smoke Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(cfg.RANDOM_SEED)

    print("[kelly] Running smoke test...")

    # Dummy policy actions
    def rand_softmax(n):
        x = np.random.randn(n)
        return np.exp(x) / np.exp(x).sum()

    action_A = rand_softmax(cfg.N_ASSETS)
    action_B = rand_softmax(cfg.N_ASSETS)
    action_C = rand_softmax(cfg.N_ASSETS)
    raw_ensemble = rand_softmax(cfg.N_ASSETS)

    test_cases = [
        {'entropy': 0.1,  'sharpe':  1.5, 'label': 'High conviction'},
        {'entropy': 1.5,  'sharpe':  0.5, 'label': 'Moderate conviction'},
        {'entropy': 2.0,  'sharpe': -0.5, 'label': 'Low conviction / crisis'},
        {'entropy': 0.05, 'sharpe':  2.5, 'label': 'Maximum conviction'},
    ]

    print(f"\n{'─'*65}")
    print(f"{'Scenario':<28} {'Frac':>6} {'R.sc':>5} {'A.sc':>5} {'S.sc':>5} "
          f"{'CASH':>6}")
    print(f"{'─'*65}")

    for tc in test_cases:
        final, kf = apply_kelly(
            raw_weights        = raw_ensemble,
            transition_entropy = tc['entropy'],
            action_A           = action_A,
            action_B           = action_B,
            action_C           = action_C,
            rolling_sharpe     = tc['sharpe'],
        )
        cash_w = final[cfg.ALL_ASSETS.index(cfg.CASH)]
        print(f"{tc['label']:<28} "
              f"{kf['fraction']:>6.3f} "
              f"{kf['regime_scalar']:>5.2f} "
              f"{kf['agreement_scalar']:>5.2f} "
              f"{kf['sharpe_scalar']:>5.2f} "
              f"{cash_w:>6.1%}")
        assert abs(final.sum() - 1.0) < 1e-5, f"Weights don't sum to 1: {final.sum()}"
        assert all(final >= 0), f"Negative weights: {final}"

    print(f"{'─'*65}")

    # Test agreement scalar with identical actions (should be ~1.0)
    identical = rand_softmax(cfg.N_ASSETS)
    a_sc = agreement_scalar(identical, identical, identical)
    print(f"\nIdentical actions → agreement scalar: {a_sc:.4f} (should be 1.0)")

    print("\n✅ Kelly smoke test passed")

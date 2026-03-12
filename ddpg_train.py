# ddpg_train.py — DDPG Ensemble Policy Training for P2-ETF-ERL-ENGINE
# Trains three DDPG policies:
#   Policy A — crisis regimes    (HMM states 4, 5, 6)
#   Policy B — expansion regimes (HMM states 0, 1, 7)
#   Policy C — full dataset      (all regimes)
#
# Each policy uses:
#   State:  80-dim (TFT embedding + HMM probs + weights + Sharpe)
#   Action: 7-dim softmax (6 ETFs + CASH)
#   Reward: log excess return vs AGG - transaction cost
#
# Run on Kaggle GPU:
#   python ddpg_train.py

import os
import sys
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from loader import load_all
from features import compute_hmm_features, load_or_compute_hmm_features, FeatureScaler
from hmm_train import RegimeDetector
from environment import PortfolioEnv, ReplayBuffer, OUNoise

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[DDPG] Using device: {DEVICE}")


# ── Actor Network ──────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Policy network: state → portfolio weights.
    Output is softmax-normalised — always sums to 1, all positive.
    """

    def __init__(
        self,
        state_dim:  int = cfg.DDPG_STATE_DIM,
        action_dim: int = cfg.DDPG_ACTION_DIM,
        hidden:     list = cfg.DDPG_ACTOR_HIDDEN,
    ):
        super().__init__()

        layers = []
        in_dim = state_dim
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(0.1),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (B, state_dim)

        Returns
        -------
        weights : (B, action_dim) — softmax normalised
        """
        logits  = self.net(state)
        weights = F.softmax(logits, dim=-1)
        return weights


# ── Critic Network ─────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Q-value network: (state, action) → scalar Q-value.
    Action is concatenated after first hidden layer.
    """

    def __init__(
        self,
        state_dim:  int = cfg.DDPG_STATE_DIM,
        action_dim: int = cfg.DDPG_ACTION_DIM,
        hidden:     list = cfg.DDPG_CRITIC_HIDDEN,
    ):
        super().__init__()

        # State branch
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.ReLU(),
        )

        # Combined branch (state + action)
        combined_dim = hidden[0] + action_dim
        layers = []
        in_dim = combined_dim
        for h in hidden[1:]:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.combined_branch = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        state:  torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        state  : (B, state_dim)
        action : (B, action_dim)

        Returns
        -------
        q_value : (B, 1)
        """
        s_feat   = self.state_branch(state)
        combined = torch.cat([s_feat, action], dim=-1)
        q_value  = self.combined_branch(combined)
        return q_value


# ── DDPG Agent ─────────────────────────────────────────────────────────────────

class DDPGAgent:
    """
    DDPG agent with:
    - Actor + target actor
    - Critic + target critic
    - Soft target updates (Polyak averaging)
    - Ornstein-Uhlenbeck exploration noise
    - Experience replay
    """

    def __init__(self, policy_name: str = 'C'):
        self.policy_name = policy_name

        # Networks
        self.actor         = Actor().to(DEVICE)
        self.actor_target  = copy.deepcopy(self.actor).to(DEVICE)
        self.critic        = Critic().to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic).to(DEVICE)

        # Optimisers
        self.actor_optim  = optim.Adam(
            self.actor.parameters(),  lr=cfg.DDPG_LR_ACTOR
        )
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=cfg.DDPG_LR_CRITIC
        )

        # Replay + noise
        self.buffer = ReplayBuffer(cfg.DDPG_BUFFER_SIZE)
        self.noise  = OUNoise(cfg.DDPG_ACTION_DIM)

        # Freeze target networks
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

    def select_action(
        self,
        state:    np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """Select action with optional OU exploration noise."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        self.actor.train()

        if training:
            noise  = self.noise.sample()
            action = action + noise
            # Re-normalise after noise addition
            action = np.exp(action) / np.exp(action).sum()

        return action.astype(np.float32)

    def update(self) -> tuple[float, float]:
        """
        Sample from replay buffer and update actor + critic.

        Returns
        -------
        (critic_loss, actor_loss) as floats
        """
        if not self.buffer.ready:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(cfg.DDPG_BATCH_SIZE)

        states_t      = torch.FloatTensor(states).to(DEVICE)
        actions_t     = torch.FloatTensor(actions).to(DEVICE)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_states_t = torch.FloatTensor(next_states).to(DEVICE)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

        # ── Critic update ─────────────────────────────────────────────────
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q     = self.critic_target(next_states_t, next_actions)
            target_q     = rewards_t + cfg.DDPG_GAMMA * target_q * (1 - dones_t)

        current_q    = self.critic(states_t, actions_t)
        critic_loss  = F.mse_loss(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # ── Actor update ──────────────────────────────────────────────────
        pred_actions = self.actor(states_t)
        actor_loss   = -self.critic(states_t, pred_actions).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ── Soft target updates ───────────────────────────────────────────
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return float(critic_loss), float(actor_loss)

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Polyak averaging: θ_target ← τ*θ + (1-τ)*θ_target"""
        tau = cfg.DDPG_TAU
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(tau * src_p.data + (1 - tau) * tgt_p.data)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) \
            if os.path.dirname(path) else None
        torch.save({
            'actor':          self.actor.state_dict(),
            'actor_target':   self.actor_target.state_dict(),
            'critic':         self.critic.state_dict(),
            'critic_target':  self.critic_target.state_dict(),
            'policy_name':    self.policy_name,
            'saved_at':       datetime.utcnow().isoformat(),
        }, path)
        print(f"[DDPG] Policy {self.policy_name} saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        print(f"[DDPG] Policy {self.policy_name} loaded ← {path}")


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_policy(
    agent:    DDPGAgent,
    env:      PortfolioEnv,
    val_env:  PortfolioEnv,
) -> dict:
    """
    Train a DDPG policy with early stopping on validation performance.

    Returns
    -------
    dict: training log
    """
    best_val_return = -np.inf
    best_state      = None
    patience_count  = 0
    train_log       = []

    print(f"\n[DDPG] Training Policy {agent.policy_name} "
          f"for up to {cfg.DDPG_MAX_EPOCHS} epochs...")

    for epoch in range(1, cfg.DDPG_MAX_EPOCHS + 1):

        # ── Training episode ──────────────────────────────────────────────
        state    = env.reset()
        done     = False
        ep_reward = 0.0
        c_losses  = []
        a_losses  = []

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)

            c_loss, a_loss = agent.update()
            c_losses.append(c_loss)
            a_losses.append(a_loss)

            ep_reward += reward
            state      = next_state

        agent.noise.decay()

        train_return = env.total_return()
        train_sharpe = env.sharpe_ratio()

        # ── Validation episode ────────────────────────────────────────────
        val_return, val_sharpe = evaluate_policy(agent, val_env)

        train_log.append({
            'epoch':        epoch,
            'train_return': float(train_return),
            'train_sharpe': float(train_sharpe),
            'val_return':   float(val_return),
            'val_sharpe':   float(val_sharpe),
            'critic_loss':  float(np.mean(c_losses)) if c_losses else 0.0,
            'actor_loss':   float(np.mean(a_losses))  if a_losses else 0.0,
            'ep_reward':    float(ep_reward),
        })

        print(f"[DDPG-{agent.policy_name}] Epoch {epoch:3d} | "
              f"Train: {train_return:.3%} | Val: {val_return:.3%} | "
              f"Sharpe: {val_sharpe:.3f}")

        # Early stopping on validation return
        if val_return > best_val_return + 1e-4:
            best_val_return = val_return
            best_state      = {
                k: v.cpu().clone()
                for k, v in agent.actor.state_dict().items()
            }
            patience_count  = 0
        else:
            patience_count += 1
            if patience_count >= cfg.DDPG_EARLY_STOP_PAT:
                print(f"[DDPG-{agent.policy_name}] Early stop "
                      f"at epoch {epoch}")
                break

    # Restore best actor weights
    if best_state:
        agent.actor.load_state_dict(best_state)
        print(f"[DDPG-{agent.policy_name}] Restored best weights "
              f"(val_return={best_val_return:.3%})")

    return {
        'best_val_return': float(best_val_return),
        'best_epoch':      int(np.argmax(
                               [r['val_return'] for r in train_log]
                           ) + 1),
        'log':             train_log,
    }


def evaluate_policy(
    agent: DDPGAgent,
    env:   PortfolioEnv,
) -> tuple[float, float]:
    """
    Run one greedy evaluation episode.

    Returns
    -------
    (total_return, sharpe_ratio)
    """
    agent.actor.eval()
    state = env.reset()
    done  = False

    while not done:
        action = agent.select_action(state, training=False)
        state, _, done, _ = env.step(action)

    agent.actor.train()
    return env.total_return(), env.sharpe_ratio()


# ── Data Preparation ───────────────────────────────────────────────────────────

def load_training_data() -> dict:
    """
    Load all data needed for policy training.
    Downloads fresh from HF.
    """
    print("[DDPG] Loading data...")
    data = load_all()

    # Load TFT feature cache (pre-computed embeddings)
    print("[DDPG] Loading TFT embeddings...")
    try:
        cache_path = hf_hub_download(
            repo_id     = cfg.HF_MODELS_REPO,
            filename    = cfg.FEATURE_CACHE_PATH,
            repo_type   = "dataset",
            token       = cfg.HF_TOKEN,
            force_download = True,
        )
        embeddings = pd.read_parquet(cache_path)
        embeddings.index = pd.to_datetime(embeddings.index)
        # Align embeddings to returns index
        common = embeddings.index.intersection(data['returns'].index)
        embeddings = embeddings.reindex(common)
        print(f"[DDPG] TFT embeddings loaded: {embeddings.shape}, common days: {len(common)}")
    except Exception as e:
        print(f"[DDPG] TFT cache not found ({e}) — using random embeddings")
        idx = data['returns'].index
        embeddings = pd.DataFrame(
            np.random.randn(len(idx), cfg.TFT_EMBEDDING_DIM) * 0.1,
            index   = idx,
            columns = [f'emb_{i}' for i in range(cfg.TFT_EMBEDDING_DIM)],
        )

    # Load HMM regime labels + probabilities
    print("[DDPG] Loading HMM regime data...")
    try:
        det_path = hf_hub_download(
            repo_id     = cfg.HF_MODELS_REPO,
            filename    = "models/regime_detector.pkl",
            repo_type   = "dataset",
            token       = cfg.HF_TOKEN,
            force_download = True,
        )
        detector  = RegimeDetector.load(det_path)
        hmm_feats = load_or_compute_hmm_features(data)
        scaler    = detector.scaler
        X_all     = scaler.transform(hmm_feats).values
        labels    = pd.Series(
            detector.model.predict(X_all),
            index = hmm_feats.index,
            name  = 'regime',
        )
        probs_arr = detector.model.predict_proba(X_all)
        hmm_probs = pd.DataFrame(
            probs_arr,
            index   = hmm_feats.index,
            columns = list(range(cfg.HMM_N_STATES)),
        )
        # Align to returns index
        common2 = hmm_probs.index.intersection(data['returns'].index)
        hmm_probs = hmm_probs.reindex(common2)
        labels    = labels.reindex(common2)
        print(f"[DDPG] HMM labels loaded: {len(labels)} days")
    except Exception as e:
        print(f"[DDPG] HMM not found ({e}) — using uniform probs")
        idx       = data['returns'].index
        labels    = pd.Series(0, index=idx, name='regime')
        hmm_probs = pd.DataFrame(
            np.ones((len(idx), cfg.HMM_N_STATES)) / cfg.HMM_N_STATES,
            index   = idx,
            columns = list(range(cfg.HMM_N_STATES)),
        )

    return {
        **data,
        'embeddings': embeddings,
        'hmm_probs':  hmm_probs,
        'labels':     labels,
    }


def build_env(
    data:        dict,
    start_date:  str,
    end_date:    str,
    regime_filter: list = None,
) -> PortfolioEnv:
    """
    Build a PortfolioEnv for a specific date range and optional regime filter.
    """
    base_idx = data['returns'].index
    mask = (
        (base_idx >= start_date) &
        (base_idx <= end_date)
    )
    if regime_filter is not None:
        regime_mask = data['labels'].isin(regime_filter)
        regime_mask = regime_mask.reindex(base_idx).fillna(False)
        mask        = mask & regime_mask

    idx = base_idx[mask]
    if len(idx) < cfg.TFT_CONTEXT_LENGTH + 10:
        # Not enough regime-specific data — fall back to full date range
        print(f"[DDPG] Regime filter too sparse "
              f"({len(idx)} days) — using full period")
        mask = (
            (base_idx >= start_date) &
            (base_idx <= end_date)
        )
        idx = base_idx[mask]

    # Drop dates where embeddings are NaN (start later than returns)
    idx = idx.intersection(data['embeddings'].dropna().index)

    env = PortfolioEnv(
        embeddings    = data['embeddings'].reindex(idx),
        hmm_probs     = data['hmm_probs'].reindex(idx),
        asset_returns = data['returns'].reindex(idx),
        bench_returns = data['bench_returns'].reindex(idx),
    )
    return env


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
    data = load_training_data()

    # ── 2. Define training periods ────────────────────────────────────────────
    # Train: 2008 → 2022   Val: 2023 → TRAIN_END
    TRAIN_START = cfg.DATA_START
    TRAIN_STOP  = '2022-12-31'
    VAL_START   = '2023-01-01'
    VAL_STOP    = cfg.TRAIN_END

    all_logs = {}

    # ── 3. Train Policy C (full dataset — train first, others build on it) ────
    policies = [
        ('C', None,                   cfg.POLICY_C_PATH),
        ('A', cfg.POLICY_A_REGIMES,   cfg.POLICY_A_PATH),
        ('B', cfg.POLICY_B_REGIMES,   cfg.POLICY_B_PATH),
    ]

    for policy_name, regime_filter, hf_path in policies:
        print(f"\n{'='*60}")
        print(f"Training Policy {policy_name}")
        if regime_filter:
            names = [cfg.REGIME_NAMES.get(r, str(r)) for r in regime_filter]
            print(f"Regimes: {names}")
        else:
            print("Regimes: All")
        print(f"{'='*60}")

        # Build environments
        train_env = build_env(
            data, TRAIN_START, TRAIN_STOP, regime_filter
        )
        val_env   = build_env(
            data, VAL_START, VAL_STOP, regime_filter
        )

        print(f"[DDPG-{policy_name}] Train env: {train_env.T} steps | "
              f"Val env: {val_env.T} steps")

        # Initialise agent
        # For A and B: warm-start from Policy C weights
        agent = DDPGAgent(policy_name=policy_name)
        if policy_name in ('A', 'B'):
            c_path = os.path.join(cfg.LOCAL_TMP, "policy_C.pt")
            if os.path.exists(c_path):
                try:
                    ckpt = torch.load(c_path, map_location=DEVICE)
                    agent.actor.load_state_dict(ckpt['actor'])
                    agent.actor_target.load_state_dict(ckpt['actor_target'])
                    agent.critic.load_state_dict(ckpt['critic'])
                    agent.critic_target.load_state_dict(ckpt['critic_target'])
                    print(f"[DDPG-{policy_name}] Warm-started from Policy C")
                except Exception as e:
                    print(f"[DDPG-{policy_name}] Could not load Policy C: {e}")

        # Train
        log = train_policy(agent, train_env, val_env)
        all_logs[policy_name] = log

        # Final evaluation
        final_return, final_sharpe = evaluate_policy(agent, val_env)
        print(f"\n[DDPG-{policy_name}] Final val return: {final_return:.3%} | "
              f"Sharpe: {final_sharpe:.3f}")

        # ── Save locally ──────────────────────────────────────────────────
        local_path = os.path.join(
            cfg.LOCAL_TMP, f"policy_{policy_name}.pt"
        )
        agent.save(local_path)

        log_path = os.path.join(
            cfg.LOCAL_TMP, f"policy_{policy_name}_log.json"
        )
        with open(log_path, 'w') as f:
            json.dump({
                'policy':         policy_name,
                'regime_filter':  regime_filter,
                'train_start':    TRAIN_START,
                'train_stop':     TRAIN_STOP,
                'val_start':      VAL_START,
                'val_stop':       VAL_STOP,
                'final_return':   float(final_return),
                'final_sharpe':   float(final_sharpe),
                'best_val_return':log['best_val_return'],
                'best_epoch':     log['best_epoch'],
                'trained_at':     datetime.utcnow().isoformat(),
            }, f, indent=2)

        # ── Push to HuggingFace ───────────────────────────────────────────
        push_to_hf(local_path, cfg.HF_MODELS_REPO, hf_path)
        push_to_hf(log_path,   cfg.HF_MODELS_REPO,
                   f"models/policy_{policy_name}_log.json")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DDPG Training Complete")
    print(f"{'='*60}")
    for name, log in all_logs.items():
        print(f"  Policy {name}: best_val={log['best_val_return']:.3%} "
              f"@ epoch {log['best_epoch']}")

    summary_path = os.path.join(cfg.LOCAL_TMP, "ddpg_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'trained_at': datetime.utcnow().isoformat(),
            'policies':   {
                name: {
                    'best_val_return': log['best_val_return'],
                    'best_epoch':      log['best_epoch'],
                }
                for name, log in all_logs.items()
            }
        }, f, indent=2)
    push_to_hf(summary_path, cfg.HF_MODELS_REPO,
               "models/ddpg_summary.json")

    print("\n[DDPG] ✅ All three policies pushed to HuggingFace")


if __name__ == "__main__":
    main()

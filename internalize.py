# internalize.py — SFT Distillation for P2-ETF-ERL-ENGINE
# Distills successful second-attempt behaviours back into the base policy
# so the model no longer needs reflection at inference time.
#
# Process:
#   1. Collect (state, action) pairs from successful second attempts
#      (those that improved by >= 50bps vs first attempt)
#   2. Supervised fine-tune the actor to reproduce those actions
#   3. Save updated actor weights back to the policy checkpoints
#
# Called at the end of erl_train.py (or separately):
#   python internalize.py

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from ddpg_train import DDPGAgent
from memory import Rulebook

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[SFT] Using device: {DEVICE}")


# ── Demonstration Dataset ──────────────────────────────────────────────────────

class DemonstrationDataset(Dataset):
    """
    Dataset of (state, action) pairs from successful second attempts.
    Actions are the reflection-guided behaviours we want to distill.
    """

    def __init__(
        self,
        states:  np.ndarray,   # (N, DDPG_STATE_DIM)
        actions: np.ndarray,   # (N, N_ASSETS)
    ):
        assert len(states) == len(actions)
        self.states  = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# ── SFT Trainer ────────────────────────────────────────────────────────────────

class SFTTrainer:
    """
    Supervised fine-tuning of the actor network.
    Uses KL divergence loss — the actor output is a softmax distribution,
    so we treat the demonstration actions as target distributions.
    """

    def __init__(
        self,
        agent:       DDPGAgent,
        lr:          float = 5e-5,    # much lower LR than DDPG — conservative
        max_epochs:  int   = 20,
        batch_size:  int   = 32,
        kl_weight:   float = 1.0,
        reg_weight:  float = 0.1,     # L2 regularisation — prevent forgetting
    ):
        self.agent       = agent
        self.lr          = lr
        self.max_epochs  = max_epochs
        self.batch_size  = batch_size
        self.kl_weight   = kl_weight
        self.reg_weight  = reg_weight

        # Snapshot original weights for regularisation
        self.original_weights = {
            k: v.clone().detach()
            for k, v in agent.actor.named_parameters()
        }

        self.optimizer = optim.Adam(agent.actor.parameters(), lr=lr)

    def _kl_loss(
        self,
        pred_logits: torch.Tensor,   # (B, N_ASSETS) raw logits
        target:      torch.Tensor,   # (B, N_ASSETS) target weights (sum to 1)
    ) -> torch.Tensor:
        """
        KL divergence: KL(target || predicted).
        Measures how much the actor's distribution diverges from the
        demonstration distribution.
        """
        pred_log = torch.log_softmax(pred_logits, dim=-1)
        target   = target.clamp(min=1e-8)
        target   = target / target.sum(dim=-1, keepdim=True)
        kl       = (target * (torch.log(target) - pred_log)).sum(dim=-1)
        return kl.mean()

    def _forgetting_penalty(self) -> torch.Tensor:
        """
        L2 penalty to prevent catastrophic forgetting of DDPG knowledge.
        Penalises large deviations from pre-SFT actor weights.
        """
        penalty = torch.tensor(0.0, device=DEVICE)
        for name, param in self.agent.actor.named_parameters():
            orig = self.original_weights[name].to(DEVICE)
            penalty += ((param - orig) ** 2).sum()
        return penalty

    def train(
        self,
        dataset:   DemonstrationDataset,
        val_split: float = 0.15,
    ) -> dict:
        """
        Fine-tune actor on demonstration dataset.

        Returns
        -------
        dict: training log
        """
        n_val   = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val

        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.RANDOM_SEED)
        )

        train_loader = DataLoader(
            train_ds,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = 0,
        )

        print(f"[SFT] Training on {n_train} demos, validating on {n_val}")

        best_val_loss = np.inf
        best_state    = None
        log           = []

        for epoch in range(1, self.max_epochs + 1):

            # ── Train ─────────────────────────────────────────────────────
            self.agent.actor.train()
            train_losses = []

            for states_b, actions_b in train_loader:
                states_b  = states_b.to(DEVICE)
                actions_b = actions_b.to(DEVICE)

                # Forward pass — get raw logits before softmax
                # Temporarily access net directly for logits
                logits = self.agent.actor.net(states_b)

                kl_loss  = self._kl_loss(logits, actions_b)
                reg_loss = self._forgetting_penalty()
                loss     = (
                    self.kl_weight  * kl_loss +
                    self.reg_weight * reg_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.actor.parameters(), max_norm=0.5
                )
                self.optimizer.step()
                train_losses.append(float(kl_loss))

            # ── Validate ──────────────────────────────────────────────────
            self.agent.actor.eval()
            val_losses = []
            with torch.no_grad():
                for states_b, actions_b in val_loader:
                    states_b  = states_b.to(DEVICE)
                    actions_b = actions_b.to(DEVICE)
                    logits    = self.agent.actor.net(states_b)
                    val_losses.append(float(self._kl_loss(logits, actions_b)))

            train_loss = np.mean(train_losses)
            val_loss   = np.mean(val_losses)

            log.append({
                'epoch':      epoch,
                'train_kl':   float(train_loss),
                'val_kl':     float(val_loss),
            })

            print(f"[SFT] Epoch {epoch:3d} | "
                  f"Train KL: {train_loss:.4f} | Val KL: {val_loss:.4f}")

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state    = {
                    k: v.cpu().clone()
                    for k, v in self.agent.actor.state_dict().items()
                }

        if best_state:
            self.agent.actor.load_state_dict(best_state)
            print(f"[SFT] Best val KL: {best_val_loss:.4f}")

        return {
            'best_val_kl': float(best_val_loss),
            'log':         log,
            'n_demos':     len(dataset),
        }


# ── Demonstration Collector ────────────────────────────────────────────────────

class DemoCollector:
    """
    Collects (state, action) demonstrations from ERL episodes.
    Only stores pairs from successful second attempts (>= 50bps improvement).
    """

    def __init__(self):
        self.states_A:  list = []
        self.actions_A: list = []
        self.states_B:  list = []
        self.actions_B: list = []
        self.states_C:  list = []
        self.actions_C: list = []

    def add(
        self,
        states:   list,
        actions:  list,
        regime_id: int,
    ):
        """Add state-action pairs from a successful second attempt."""
        if not states or not actions:
            return

        s_arr = np.array(states,  dtype=np.float32)
        a_arr = np.array(actions, dtype=np.float32)

        # Route to policy bucket
        if regime_id in cfg.POLICY_A_REGIMES:
            self.states_A.extend(s_arr)
            self.actions_A.extend(a_arr)
        if regime_id in cfg.POLICY_B_REGIMES:
            self.states_B.extend(s_arr)
            self.actions_B.extend(a_arr)

        # All demonstrations go to Policy C
        self.states_C.extend(s_arr)
        self.actions_C.extend(a_arr)

    def get_dataset(self, policy: str) -> DemonstrationDataset | None:
        """Build a DemonstrationDataset for the given policy."""
        mapping = {
            'A': (self.states_A, self.actions_A),
            'B': (self.states_B, self.actions_B),
            'C': (self.states_C, self.actions_C),
        }
        states, actions = mapping[policy]
        if len(states) < 10:
            print(f"[SFT] Policy {policy}: insufficient demos "
                  f"({len(states)}) — skipping")
            return None
        return DemonstrationDataset(
            np.array(states),
            np.array(actions),
        )

    def sizes(self) -> dict:
        return {
            'A': len(self.states_A),
            'B': len(self.states_B),
            'C': len(self.states_C),
        }


# ── HF Push ────────────────────────────────────────────────────────────────────

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

def internalize(
    agents:     dict,
    collector:  DemoCollector,
) -> dict:
    """
    Run SFT distillation for all policies that have enough demos.
    Called from erl_train.py after the main ERL loop.

    Parameters
    ----------
    agents    : dict {'A': DDPGAgent, 'B': ..., 'C': ...}
    collector : DemoCollector with accumulated demonstrations

    Returns
    -------
    dict: SFT results per policy
    """
    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)
    results = {}
    sizes   = collector.sizes()

    print(f"\n[SFT] Demo sizes: {sizes}")

    hf_paths = {
        'A': cfg.POLICY_A_PATH,
        'B': cfg.POLICY_B_PATH,
        'C': cfg.POLICY_C_PATH,
    }

    for name, agent in agents.items():
        dataset = collector.get_dataset(name)
        if dataset is None:
            results[name] = {'skipped': True, 'n_demos': sizes[name]}
            continue

        print(f"\n[SFT] Distilling Policy {name} "
              f"({len(dataset)} demonstrations)...")

        trainer = SFTTrainer(agent)
        log     = trainer.train(dataset)
        results[name] = log

        # Save updated policy
        local_path = os.path.join(cfg.LOCAL_TMP, f"policy_{name}.pt")
        agent.save(local_path)
        push_to_hf(local_path, cfg.HF_MODELS_REPO, hf_paths[name])

    # Save SFT summary
    summary_path = os.path.join(cfg.LOCAL_TMP, "sft_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'internalized_at': datetime.utcnow().isoformat(),
            'demo_sizes':      sizes,
            'results':         {
                k: {kk: vv for kk, vv in v.items() if kk != 'log'}
                for k, v in results.items()
            },
        }, f, indent=2)
    push_to_hf(summary_path, cfg.HF_MODELS_REPO, "models/sft_summary.json")

    print(f"\n[SFT] ✅ Internalization complete")
    for name, res in results.items():
        if res.get('skipped'):
            print(f"  Policy {name}: skipped ({res['n_demos']} demos)")
        else:
            print(f"  Policy {name}: best_val_kl={res.get('best_val_kl', '?'):.4f}, "
                  f"n_demos={res.get('n_demos', '?')}")

    return results


# ── Standalone Entry Point ─────────────────────────────────────────────────────

def main():
    """
    Standalone mode: load policies + a saved demo buffer and run SFT.
    Useful for re-running internalization without re-running the full ERL loop.
    """
    cfg.validate()
    torch.manual_seed(cfg.RANDOM_SEED)

    # Load agents
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
        except Exception as e:
            print(f"[SFT] Could not load Policy {name}: {e}")

    if not agents:
        print("[SFT] No policies loaded — nothing to internalize")
        return

    # Try to load saved demo buffer
    demo_path = os.path.join(cfg.LOCAL_TMP, "demo_buffer.json")
    if not os.path.exists(demo_path):
        print(f"[SFT] No demo buffer at {demo_path} — "
              f"run erl_train.py first to generate demonstrations")
        return

    with open(demo_path, 'r') as f:
        demo_data = json.load(f)

    collector = DemoCollector()
    for entry in demo_data:
        collector.add(
            states    = entry['states'],
            actions   = entry['actions'],
            regime_id = entry['regime_id'],
        )

    internalize(agents, collector)


if __name__ == "__main__":
    main()

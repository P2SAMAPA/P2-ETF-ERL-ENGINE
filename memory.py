# memory.py — Cross-Episode Rulebook for P2-ETF-ERL-ENGINE
# Stores, retrieves, and manages the rolling rulebook of successful
# reflections. Acts as the long-term memory of the ERL loop.
#
# Design:
#   - Rolling window of last MAX_RULES reflections (default 20)
#   - Indexed by regime for fast retrieval
#   - Persisted to HF_RESULTS_REPO after each update
#   - Loaded at start of each training session
#
# Used by: erl_train.py, reflect.py, predict.py

import os
import sys
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Optional
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


# ── Rulebook ───────────────────────────────────────────────────────────────────

class Rulebook:
    """
    Rolling rulebook of successful ERL reflections.

    Structure:
        rules : list of rule dicts, ordered by insertion time
                Each rule has: regime_id, regime_name, rule_type,
                primary_asset, secondary_asset, condition, action,
                rationale, confidence, source, generated_at,
                improvement, episode_id

    Max size: cfg.ERL_MAX_RULES (20)
    When full: oldest rule is evicted (FIFO with regime diversity preference)
    """

    def __init__(self):
        self.rules: list[dict]      = []
        self.episode_count: int     = 0
        self.total_stored: int      = 0
        self.total_rejected: int    = 0
        self.created_at: str        = datetime.utcnow().isoformat()
        self.last_updated: str      = self.created_at

    # ── Storage ────────────────────────────────────────────────────────────────

    def add(self, rule: dict, improvement: float, episode_id: int) -> bool:
        """
        Attempt to add a rule to the rulebook.
        Enforces the 50bps gate — rejects rules below threshold.

        Parameters
        ----------
        rule        : dict from Reflector.generate_reflection()
        improvement : float — excess return improvement from 1st to 2nd attempt
        episode_id  : int

        Returns
        -------
        bool: True if rule was stored, False if rejected
        """
        # Gate check
        if improvement < cfg.ERL_MIN_EXCESS_TO_STORE:
            self.total_rejected += 1
            return False

        # Deduplicate — reject if identical action+regime already stored
        action     = rule.get('action', '')
        regime_id  = rule.get('regime_id', -1)
        if any(r.get('action') == action and r.get('regime_id') == regime_id
               for r in self.rules):
            return False  # duplicate rule, skip

        # Enrich rule with storage metadata
        enriched = {
            **rule,
            'improvement': float(improvement),
            'episode_id':  episode_id,
            'stored_at':   datetime.utcnow().isoformat(),
        }

        # Evict if at capacity — prefer evicting oldest rule from
        # most-represented regime to maintain regime diversity
        if len(self.rules) >= cfg.ERL_MAX_RULES:
            self._evict_one()

        self.rules.append(enriched)
        self.total_stored += 1
        self.last_updated = datetime.utcnow().isoformat()
        return True

    def _evict_one(self):
        """
        Evict one rule. Strategy:
        1. Find the regime with the most rules
        2. Remove the oldest rule from that regime
        This maintains diversity across regimes.
        """
        if not self.rules:
            return

        # Count rules per regime
        regime_counts = defaultdict(list)
        for i, r in enumerate(self.rules):
            regime_counts[r['regime_id']].append(i)

        # Find most-represented regime
        most_common_regime = max(
            regime_counts, key=lambda k: len(regime_counts[k])
        )

        # Remove the oldest rule from that regime
        oldest_idx = regime_counts[most_common_regime][0]
        evicted    = self.rules.pop(oldest_idx)

        print(f"[memory] Evicted rule: regime={evicted['regime_name']}, "
              f"episode={evicted.get('episode_id', '?')}")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def get_rules_for_regime(
        self,
        regime_id: int,
        max_rules: int = 5,
    ) -> list[dict]:
        """
        Retrieve rules relevant to a given regime.
        Returns exact regime matches first, then adjacent regimes.

        Parameters
        ----------
        regime_id : int — current HMM regime label
        max_rules : int — maximum rules to return

        Returns
        -------
        list of rule dicts, sorted by confidence desc
        """
        # Exact matches
        exact = [
            r for r in self.rules
            if r['regime_id'] == regime_id
        ]

        # Adjacent regime rules (same policy bucket)
        policy_bucket = self._get_policy_bucket(regime_id)
        adjacent = [
            r for r in self.rules
            if r['regime_id'] != regime_id and
               self._get_policy_bucket(r['regime_id']) == policy_bucket
        ]

        # Combine: exact first, then adjacent, sorted by confidence
        combined = (
            sorted(exact,    key=lambda r: -r.get('confidence', 0.5)) +
            sorted(adjacent, key=lambda r: -r.get('confidence', 0.5))
        )
        return combined[:max_rules]

    def get_all_rules(self) -> list[dict]:
        """Return all rules sorted by confidence desc."""
        return sorted(self.rules, key=lambda r: -r.get('confidence', 0.5))

    def get_regime_summary(self) -> dict:
        """Return count of rules per regime."""
        counts = defaultdict(int)
        for r in self.rules:
            counts[cfg.REGIME_NAMES.get(r['regime_id'],
                                        f"Regime {r['regime_id']}")] += 1
        return dict(counts)

    @staticmethod
    def _get_policy_bucket(regime_id: int) -> str:
        """Map regime to policy bucket for adjacency matching."""
        if regime_id in cfg.POLICY_A_REGIMES:
            return 'crisis'
        elif regime_id in cfg.POLICY_B_REGIMES:
            return 'expansion'
        return 'neutral'

    # ── Persistence ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            'rules':            self.rules,
            'episode_count':    self.episode_count,
            'total_stored':     self.total_stored,
            'total_rejected':   self.total_rejected,
            'created_at':       self.created_at,
            'last_updated':     self.last_updated,
            'n_rules':          len(self.rules),
            'regime_summary':   self.get_regime_summary(),
        }

    def save_local(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) \
            if os.path.dirname(path) else None
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[memory] Rulebook saved → {path} "
              f"({len(self.rules)} rules)")

    def push_to_hf(self, local_path: str = None):
        """Push rulebook to HF_RESULTS_REPO."""
        if local_path is None:
            local_path = os.path.join(cfg.LOCAL_TMP, "rulebook.json")
        self.save_local(local_path)

        api = HfApi(token=cfg.HF_TOKEN)
        api.upload_file(
            path_or_fileobj = local_path,
            path_in_repo    = cfg.RULEBOOK_PATH,
            repo_id         = cfg.HF_RESULTS_REPO,
            repo_type       = "dataset",
        )
        print(f"[memory] Rulebook pushed → {cfg.HF_RESULTS_REPO} "
              f"({len(self.rules)} rules)")

    @staticmethod
    def load_local(path: str) -> 'Rulebook':
        rb = Rulebook()
        with open(path, 'r') as f:
            data = json.load(f)
        rb.rules          = data.get('rules', [])
        rb.episode_count  = data.get('episode_count', 0)
        rb.total_stored   = data.get('total_stored', 0)
        rb.total_rejected = data.get('total_rejected', 0)
        rb.created_at     = data.get('created_at', rb.created_at)
        rb.last_updated   = data.get('last_updated', rb.last_updated)
        print(f"[memory] Rulebook loaded ← {path} "
              f"({len(rb.rules)} rules)")
        return rb

    @staticmethod
    def load_from_hf() -> 'Rulebook':
        """
        Download rulebook from HF_RESULTS_REPO.
        Returns empty rulebook if not found.
        """
        try:
            path = hf_hub_download(
                repo_id      = cfg.HF_RESULTS_REPO,
                filename     = cfg.RULEBOOK_PATH,
                repo_type    = "dataset",
                token        = cfg.HF_TOKEN,
                force_download = True,
            )
            return Rulebook.load_local(path)
        except Exception as e:
            print(f"[memory] No existing rulebook found ({e}) "
                  f"— starting fresh")
            return Rulebook()

    # ── Display ────────────────────────────────────────────────────────────────

    def print_summary(self):
        print(f"\n── Rulebook Summary ──────────────────────────────")
        print(f"  Total rules:    {len(self.rules)} / {cfg.ERL_MAX_RULES}")
        print(f"  Total stored:   {self.total_stored}")
        print(f"  Total rejected: {self.total_rejected} "
              f"(below {cfg.ERL_MIN_EXCESS_TO_STORE:.1%} gate)")
        print(f"  Last updated:   {self.last_updated}")

        if self.rules:
            print(f"\n  Per-regime distribution:")
            for regime_name, count in self.get_regime_summary().items():
                print(f"    {regime_name:25s}: {count} rule(s)")

            print(f"\n  Top 5 rules by confidence:")
            for r in self.get_all_rules()[:5]:
                print(f"    [{r['regime_name']:20s}] "
                      f"{r.get('action', r.get('rationale',''))[:60]} "
                      f"(conf={r.get('confidence', 0.5):.2f}, "
                      f"impr={r.get('improvement', 0):.2%})")


# ── Audit Trail ────────────────────────────────────────────────────────────────

class AuditTrail:
    """
    Lightweight append-only log of every ERL episode outcome.
    Stored in HF_RESULTS_REPO for transparency and debugging.
    """

    def __init__(self):
        self.entries: list[dict] = []

    def record(
        self,
        episode_id:    int,
        regime_id:     int,
        regime_name:   str,
        first_excess:  float,
        second_excess: float,
        reflection:    Optional[dict],
        stored:        bool,
    ):
        self.entries.append({
            'episode_id':    episode_id,
            'regime_id':     regime_id,
            'regime_name':   regime_name,
            'first_excess':  float(first_excess),
            'second_excess': float(second_excess) if second_excess is not None else None,
            'improvement':   float(second_excess - first_excess)
                             if second_excess is not None else None,
            'rule_type':     reflection.get('rule_type')
                             if reflection else None,
            'stored':        stored,
            'source':        reflection.get('source', 'unknown')
                             if reflection else None,
            'recorded_at':   datetime.utcnow().isoformat(),
        })

    def save_local(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) \
            if os.path.dirname(path) else None
        with open(path, 'w') as f:
            json.dump(self.entries, f, indent=2)

    def push_to_hf(self, local_path: str = None):
        if local_path is None:
            local_path = os.path.join(cfg.LOCAL_TMP, "audit_trail.json")
        self.save_local(local_path)
        api = HfApi(token=cfg.HF_TOKEN)
        api.upload_file(
            path_or_fileobj = local_path,
            path_in_repo    = "results/audit_trail.json",
            repo_id         = cfg.HF_RESULTS_REPO,
            repo_type       = "dataset",
        )
        print(f"[memory] Audit trail pushed "
              f"({len(self.entries)} entries)")

    def summary(self) -> dict:
        if not self.entries:
            return {'total_episodes': 0}
        improvements = [
            e['improvement'] for e in self.entries
            if e['improvement'] is not None
        ]
        stored = [e for e in self.entries if e['stored']]
        return {
            'total_episodes':    len(self.entries),
            'total_stored':      len(stored),
            'mean_improvement':  float(np.mean(improvements))
                                 if improvements else 0.0,
            'pct_improved':      float(
                sum(1 for i in improvements if i > 0) /
                len(improvements)
            ) if improvements else 0.0,
            'gemini_reflections': sum(
                1 for e in self.entries
                if e.get('source') == 'gemini'
            ),
        }


# ── Smoke Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[memory] Running smoke test...")

    rb = Rulebook()

    # Add rules for various regimes
    test_rules = [
        {
            'regime_id':       4,
            'regime_name':     'Credit Stress',
            'rule_type':       'reduce',
            'primary_asset':   'HYG',
            'secondary_asset': 'GLD',
            'condition':       'In Credit Stress with negative excess return',
            'action':          'Reduce HYG by 20%, increase GLD allocation',
            'rationale':       'HYG underperforms in credit stress; GLD holds value.',
            'confidence':      0.78,
            'source':          'gemini',
        },
        {
            'regime_id':       6,
            'regime_name':     'Acute Crisis',
            'rule_type':       'prefer',
            'primary_asset':   'TLT',
            'secondary_asset': 'CASH',
            'condition':       'Acute crisis with high vol',
            'action':          'Max TLT + CASH, exit all credit',
            'rationale':       'Flight to safety in acute crisis.',
            'confidence':      0.92,
            'source':          'gemini',
        },
        {
            'regime_id':       1,
            'regime_name':     'Mid Cycle Growth',
            'rule_type':       'increase',
            'primary_asset':   'HYG',
            'secondary_asset': 'VNQ',
            'condition':       'Mid cycle with positive momentum',
            'action':          'Increase HYG and VNQ, reduce TLT',
            'rationale':       'Risk assets outperform in mid-cycle.',
            'confidence':      0.65,
            'source':          'rule_based',
        },
    ]

    for i, rule in enumerate(test_rules):
        stored = rb.add(rule, improvement=0.01 + i * 0.005, episode_id=i + 1)
        print(f"[test] Rule {i+1} stored: {stored}")

    # Test retrieval
    crisis_rules = rb.get_rules_for_regime(regime_id=4)
    print(f"\n[test] Rules for Credit Stress (regime 4): "
          f"{len(crisis_rules)} rules")

    # Test eviction — fill to capacity
    for i in range(cfg.ERL_MAX_RULES):
        rb.add({
            'regime_id':       i % cfg.HMM_N_STATES,
            'regime_name':     cfg.REGIME_NAMES.get(i % cfg.HMM_N_STATES, f'R{i}'),
            'rule_type':       'reduce',
            'primary_asset':   'HYG',
            'secondary_asset': 'GLD',
            'condition':       'test',
            'action':          f'Test rule {i}',
            'rationale':       'test',
            'confidence':      0.5,
            'source':          'test',
        }, improvement=0.01, episode_id=100 + i)

    assert len(rb.rules) <= cfg.ERL_MAX_RULES, \
        f"Rulebook exceeded max size: {len(rb.rules)}"
    print(f"\n[test] Eviction works: {len(rb.rules)} rules "
          f"(<= {cfg.ERL_MAX_RULES})")

    rb.print_summary()

    # Test save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        tmp_path = f.name
    rb.save_local(tmp_path)
    rb2 = Rulebook.load_local(tmp_path)
    assert len(rb2.rules) == len(rb.rules)
    os.unlink(tmp_path)
    print(f"[test] Save/load round-trip OK")

    # Audit trail test
    trail = AuditTrail()
    for i in range(5):
        trail.record(
            episode_id   = i,
            regime_id    = 4,
            regime_name  = 'Credit Stress',
            first_excess = -0.02,
            second_excess= 0.005,
            reflection   = {'rule_type': 'reduce', 'source': 'gemini'},
            stored       = True,
        )
    s = trail.summary()
    print(f"\n[test] Audit trail: {s}")

    print("\n✅ Memory smoke test passed")

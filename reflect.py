# reflect.py — ERL Reflection Generator for P2-ETF-ERL-ENGINE
# Generates structured reflections from failed/suboptimal episodes.
# Uses Gemini 1.5 Flash (free tier) with rule-based fallback.
#
# The reflection converts raw episode data (regime, assets held,
# returns, attention weights) into a structured behavioural rule
# that guides the second attempt and optionally enters the rulebook.
#
# Used by: erl_train.py

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


# ── Prompt Templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert quantitative portfolio analyst specialising in 
fixed-income and real-asset ETFs. You analyse failed trading episodes and generate 
precise, actionable portfolio rules.

You are managing a portfolio of 6 ETFs: TLT (long-duration Treasuries), LQD (investment 
grade bonds), HYG (high yield bonds), VNQ (REITs), GLD (gold), SLV (silver), plus CASH.
The benchmark is AGG (US Aggregate Bond Market).

Your task: analyse why a portfolio allocation underperformed AGG and generate a specific 
corrective rule for the current market regime.

Rules must be:
- Specific to the identified regime
- Actionable (specific ETF adjustments)
- Concise (2-3 sentences maximum)
- Based on the evidence provided

Always respond in valid JSON format only. No preamble, no markdown."""

REFLECTION_PROMPT_TEMPLATE = """Analyse this failed portfolio episode and generate a corrective rule.

EPISODE DATA:
- Market Regime: {regime_name} (Regime {regime_id})
- Regime Transition Risk: {transition_risk}
- Assets Allocated: {asset_allocation}
- Portfolio Return: {portfolio_return}
- Benchmark (AGG) Return: {bench_return}
- Excess Return: {excess_return} (UNDERPERFORMED)
- Episode Duration: {n_days} trading days
- Worst Single Day: {worst_day}

FEATURE ATTENTION (which historical periods the model focused on):
{attention_summary}

EXISTING RULES FOR THIS REGIME:
{existing_rules}

Generate a corrective rule in this exact JSON format:
{{
  "regime_id": {regime_id},
  "regime_name": "{regime_name}",
  "rule_type": "reduce|increase|avoid|prefer",
  "primary_asset": "<ETF or CASH>",
  "secondary_asset": "<ETF or null>",
  "condition": "<specific trigger condition>",
  "action": "<specific portfolio adjustment>",
  "rationale": "<1-2 sentence evidence-based explanation>",
  "confidence": <0.0-1.0>
}}"""

SECOND_ATTEMPT_PROMPT_TEMPLATE = """You are managing an ETF portfolio in {regime_name} regime.

ACTIVE RULES FOR THIS REGIME:
{active_rules}

CURRENT MARKET STATE:
- Regime transition probability to crisis: {crisis_prob:.1%}
- Regime entropy (uncertainty): {entropy:.3f}
- Recent portfolio Sharpe: {sharpe:.3f}
- Current weights: {current_weights}

Based on the active rules and current market state, recommend portfolio weight 
adjustments as a JSON object:
{{
  "reasoning": "<brief explanation>",
  "weight_adjustments": {{
    "TLT": <-1.0 to 1.0>,
    "LQD": <-1.0 to 1.0>,
    "HYG": <-1.0 to 1.0>,
    "VNQ": <-1.0 to 1.0>,
    "GLD": <-1.0 to 1.0>,
    "SLV": <-1.0 to 1.0>,
    "CASH": <-1.0 to 1.0>
  }},
  "conviction": <0.0-1.0>
}}

Adjustments are additive to current weights. They will be renormalised via softmax."""


# ── Gemini Client ──────────────────────────────────────────────────────────────

class GeminiReflector:
    """
    Gemini 1.5 Flash reflection generator.
    Free tier: 15 req/min, 1500 req/day, 1M tokens/day.
    """

    def __init__(self):
        self.api_key   = cfg.GEMINI_API_KEY
        self.model     = cfg.ERL_GEMINI_MODEL
        self.available = False

        if not self.api_key:
            print("[Reflect] GEMINI_API_KEY not set — "
                  "will use rule-based fallback")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client    = genai.GenerativeModel(
                model_name    = self.model,
                system_instruction = SYSTEM_PROMPT,
            )
            self.available = True
            print(f"[Reflect] Gemini {self.model} ready ✓")
        except ImportError:
            print("[Reflect] google-generativeai not installed — "
                  "run: pip install google-generativeai")
        except Exception as e:
            print(f"[Reflect] Gemini init failed: {e}")

    def generate(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Call Gemini API with retry logic.
        Returns raw text response or None on failure.
        """
        if not self.available:
            return None

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        'max_output_tokens': cfg.ERL_MAX_REFLECTION_TOKENS,
                        'temperature':       0.3,   # low temp for consistency
                    }
                )
                return response.text.strip()

            except Exception as e:
                wait = 2 ** attempt
                print(f"[Reflect] Gemini attempt {attempt+1} failed: {e} "
                      f"— retrying in {wait}s")
                time.sleep(wait)

        print("[Reflect] All Gemini attempts failed — using rule-based fallback")
        return None


# ── Rule-Based Fallback ────────────────────────────────────────────────────────

class RuleBasedReflector:
    """
    Deterministic reflection generator — no API calls.
    Used when Gemini is unavailable or as a fast alternative.
    Produces lower-quality but consistent reflections.
    """

    # Heuristic rules per regime type
    REGIME_HEURISTICS = {
        'crisis': {
            'reduce':   ['HYG', 'VNQ'],
            'increase': ['GLD', 'TLT', 'CASH'],
            'rationale': 'In crisis regimes, credit and real assets '
                         'typically underperform safe havens.',
        },
        'expansion': {
            'reduce':   ['TLT', 'CASH'],
            'increase': ['HYG', 'VNQ', 'SLV'],
            'rationale': 'In expansion regimes, risk assets and real '
                         'assets typically outperform duration.',
        },
        'flattening': {
            'reduce':   ['TLT', 'HYG'],
            'increase': ['GLD', 'LQD', 'CASH'],
            'rationale': 'Curve flattening hurts duration and credit; '
                         'gold and investment grade tend to hold value.',
        },
        'default': {
            'reduce':   [],
            'increase': ['GLD'],
            'rationale': 'Regime uncertain — reduce concentration '
                         'and increase defensive allocation.',
        },
    }

    def _get_heuristic(self, regime_name: str) -> dict:
        name_lower = regime_name.lower()
        if any(k in name_lower for k in ['crisis', 'risk off', 'acute']):
            return self.REGIME_HEURISTICS['crisis']
        elif any(k in name_lower for k in ['expansion', 'growth', 'recovery']):
            return self.REGIME_HEURISTICS['expansion']
        elif 'flat' in name_lower or 'late' in name_lower:
            return self.REGIME_HEURISTICS['flattening']
        return self.REGIME_HEURISTICS['default']

    def generate_reflection(self, episode_data: dict) -> dict:
        regime_id   = episode_data['regime_id']
        regime_name = episode_data['regime_name']
        allocation  = episode_data['asset_allocation']
        excess      = episode_data['excess_return']

        heuristic   = self._get_heuristic(regime_name)

        # Find worst performing held asset
        worst_asset = max(
            allocation,
            key=lambda a: allocation[a],
            default='HYG'
        ) if allocation else 'HYG'

        primary   = heuristic['reduce'][0] if heuristic['reduce'] else worst_asset
        secondary = heuristic['increase'][0] if heuristic['increase'] else 'GLD'

        return {
            'regime_id':       regime_id,
            'regime_name':     regime_name,
            'rule_type':       'reduce',
            'primary_asset':   primary,
            'secondary_asset': secondary,
            'condition':       f'In {regime_name} with negative excess return',
            'action':          (
                f'Reduce {primary} allocation by ~20%, '
                f'increase {secondary} allocation'
            ),
            'rationale':       heuristic['rationale'],
            'confidence':      0.5,
            'source':          'rule_based',
        }

    def generate_adjustment(self, state_data: dict,
                             active_rules: list) -> dict:
        """Generate weight adjustments from active rules."""
        adjustments = {a: 0.0 for a in cfg.ALL_ASSETS}
        crisis_prob = state_data.get('crisis_prob', 0.0)

        for rule in active_rules:
            if rule.get('rule_type') == 'reduce':
                asset = rule.get('primary_asset')
                if asset in adjustments:
                    adjustments[asset] -= 0.1 * rule.get('confidence', 0.5)
            if rule.get('rule_type') in ('increase', 'prefer'):
                asset = rule.get('secondary_asset')
                if asset and asset in adjustments:
                    adjustments[asset] += 0.1 * rule.get('confidence', 0.5)

        # Crisis override
        if crisis_prob > cfg.ENSEMBLE_CRISIS_THRESHOLD:
            adjustments['HYG']  -= 0.15
            adjustments['VNQ']  -= 0.10
            adjustments['GLD']  += 0.15
            adjustments['CASH'] += 0.10

        return {
            'reasoning':        'Rule-based adjustment from active rulebook',
            'weight_adjustments': adjustments,
            'conviction':       min(0.3 + len(active_rules) * 0.05, 0.8),
        }


# ── Main Reflector ─────────────────────────────────────────────────────────────

class Reflector:
    """
    Main reflection interface — tries Gemini first, falls back to rule-based.
    Used by erl_train.py during training.
    """

    def __init__(self):
        self.gemini    = GeminiReflector()
        self.fallback  = RuleBasedReflector()
        self.call_count = 0
        self.gemini_count = 0
        self.fallback_count = 0

    def generate_reflection(self, episode_data: dict) -> dict:
        """
        Generate a structured reflection from a failed episode.

        Parameters
        ----------
        episode_data : dict with keys:
            regime_id, regime_name, asset_allocation, portfolio_return,
            bench_return, excess_return, n_days, worst_day,
            attention_summary, existing_rules, transition_risk

        Returns
        -------
        dict: structured reflection rule
        """
        self.call_count += 1

        # Try Gemini first
        if self.gemini.available:
            prompt = REFLECTION_PROMPT_TEMPLATE.format(
                regime_id       = episode_data['regime_id'],
                regime_name     = episode_data['regime_name'],
                transition_risk = episode_data.get('transition_risk', 'Low'),
                asset_allocation= self._format_allocation(
                                      episode_data['asset_allocation']
                                  ),
                portfolio_return= f"{episode_data['portfolio_return']:.2%}",
                bench_return    = f"{episode_data['bench_return']:.2%}",
                excess_return   = f"{episode_data['excess_return']:.2%}",
                n_days          = episode_data.get('n_days', 1),
                worst_day       = f"{episode_data.get('worst_day', 0.0):.2%}",
                attention_summary = episode_data.get(
                    'attention_summary', 'Not available'
                ),
                existing_rules  = self._format_rules(
                                      episode_data.get('existing_rules', [])
                                  ),
            )

            raw = self.gemini.generate(prompt)
            if raw:
                parsed = self._parse_json_response(raw)
                if parsed:
                    parsed['source'] = 'gemini'
                    parsed['generated_at'] = datetime.utcnow().isoformat()
                    self.gemini_count += 1
                    return parsed

        # Fallback
        self.fallback_count += 1
        result = self.fallback.generate_reflection(episode_data)
        result['generated_at'] = datetime.utcnow().isoformat()
        return result

    def generate_second_attempt_adjustment(
        self,
        state_data:   dict,
        active_rules: list,
    ) -> dict:
        """
        Generate weight adjustments to guide the second attempt.

        Parameters
        ----------
        state_data : dict with keys:
            regime_name, crisis_prob, entropy, sharpe, current_weights
        active_rules : list of reflection dicts from rulebook

        Returns
        -------
        dict with weight_adjustments and conviction
        """
        if self.gemini.available and active_rules:
            prompt = SECOND_ATTEMPT_PROMPT_TEMPLATE.format(
                regime_name     = state_data['regime_name'],
                active_rules    = self._format_rules(active_rules),
                crisis_prob     = state_data.get('crisis_prob', 0.0),
                entropy         = state_data.get('entropy', 0.0),
                sharpe          = state_data.get('sharpe', 0.0),
                current_weights = self._format_allocation(
                                      state_data['current_weights']
                                  ),
            )
            raw = self.gemini.generate(prompt)
            if raw:
                parsed = self._parse_json_response(raw)
                if parsed and 'weight_adjustments' in parsed:
                    return parsed

        return self.fallback.generate_adjustment(state_data, active_rules)

    def _format_allocation(self, allocation: dict) -> str:
        """Format weight dict as readable string."""
        if not allocation:
            return 'Not available'
        return ', '.join(
            f"{k}: {v:.1%}" for k, v in sorted(
                allocation.items(), key=lambda x: -x[1]
            )
        )

    def _format_rules(self, rules: list) -> str:
        """Format rules list as readable string."""
        if not rules:
            return 'No existing rules for this regime.'
        lines = []
        for i, r in enumerate(rules[:5], 1):   # max 5 rules in prompt
            lines.append(
                f"{i}. [{r.get('rule_type','?').upper()}] "
                f"{r.get('action', r.get('rationale', ''))}"
            )
        return '\n'.join(lines)

    def _parse_json_response(self, raw: str) -> Optional[dict]:
        """
        Safely parse JSON from LLM response.
        Handles common formatting issues (markdown fences, trailing text).
        """
        if not raw:
            return None

        # Strip markdown code fences
        text = raw.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            text  = '\n'.join(
                line for line in lines
                if not line.startswith('```')
            )

        # Find JSON object boundaries
        start = text.find('{')
        end   = text.rfind('}')
        if start == -1 or end == -1:
            return None

        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            print(f"[Reflect] JSON parse error: {e}")
            return None

    def stats(self) -> dict:
        return {
            'total_calls':    self.call_count,
            'gemini_calls':   self.gemini_count,
            'fallback_calls': self.fallback_count,
            'gemini_rate':    (
                self.gemini_count / self.call_count
                if self.call_count > 0 else 0.0
            ),
        }


# ── Smoke Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[reflect] Running smoke test...")

    reflector = Reflector()

    # Test episode data
    episode_data = {
        'regime_id':       4,
        'regime_name':     'Credit Stress',
        'transition_risk': 'HIGH — 34% probability of transition to Acute Crisis',
        'asset_allocation': {
            'HYG':  0.45,
            'VNQ':  0.25,
            'LQD':  0.15,
            'TLT':  0.10,
            'GLD':  0.03,
            'SLV':  0.01,
            'CASH': 0.01,
        },
        'portfolio_return': -0.032,
        'bench_return':     -0.008,
        'excess_return':    -0.024,
        'n_days':           5,
        'worst_day':        -0.018,
        'attention_summary': (
            'Model attention concentrated on 2019 data (40%) '
            'and 2016 data (25%) — both recovery periods, '
            'not credit stress analogues.'
        ),
        'existing_rules':   [],
    }

    print("\n── Generating Reflection ─────────────────────")
    reflection = reflector.generate_reflection(episode_data)
    print(json.dumps(reflection, indent=2))

    # Test second attempt adjustment
    state_data = {
        'regime_name':   'Credit Stress',
        'crisis_prob':   0.34,
        'entropy':       1.82,
        'sharpe':        -0.45,
        'current_weights': {
            'HYG':  0.45,
            'VNQ':  0.25,
            'LQD':  0.15,
            'TLT':  0.10,
            'GLD':  0.03,
            'SLV':  0.01,
            'CASH': 0.01,
        },
    }

    print("\n── Generating Second Attempt Adjustment ──────")
    adjustment = reflector.generate_second_attempt_adjustment(
        state_data, [reflection]
    )
    print(json.dumps(adjustment, indent=2))

    print(f"\n── Reflector Stats ───────────────────────────")
    print(json.dumps(reflector.stats(), indent=2))

    print("\n✅ Reflect smoke test complete")

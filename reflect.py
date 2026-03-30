# reflect.py — ERL Reflection Generator for P2-ETF-ERL-ENGINE
# (full file as before, but with the SECOND_ATTEMPT_PROMPT_TEMPLATE fixed)

# ... all imports and earlier code unchanged up to the prompt templates ...

# ── Prompt Templates (with corrected escaping) ───────────────────────────────

def _asset_list_str() -> str:
    """Return a string listing all assets (including CASH) for prompts."""
    return ", ".join(cfg.ALL_ASSETS)

def _weight_adjustments_dict_str() -> str:
    """Return a string representation of the weight adjustments JSON object."""
    items = ",\n".join(f'    "{asset}": <-1.0 to 1.0>' for asset in cfg.ALL_ASSETS)
    return "{\n" + items + "\n  }"

SYSTEM_PROMPT = f"""You are an expert quantitative portfolio analyst specialising in 
fixed-income, real-asset, and equity ETFs. You analyse failed trading episodes and generate 
precise, actionable portfolio rules.

You are managing a portfolio of {len(cfg.ASSETS)} ETFs ({_asset_list_str()}), plus CASH.
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
  "weight_adjustments": {weight_adjustments_str},
  "conviction": <0.0-1.0>
}}

Adjustments are additive to current weights. They will be renormalised via softmax."""

# ... the rest of reflect.py remains unchanged (class definitions, etc.)

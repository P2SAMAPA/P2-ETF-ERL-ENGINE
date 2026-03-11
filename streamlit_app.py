# streamlit_app.py — REALM Dashboard

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

st.set_page_config(page_title="REALM · P2 ETF Engine", page_icon="⬡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stApp {
    background: #ffffff !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
}

.block-container { padding: 2rem 3rem !important; max-width: 1400px; }

/* Cards */
.kpi-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 32px;
    font-weight: 600;
    color: #111827;
    line-height: 1;
}
.kpi-value.green  { color: #059669; }
.kpi-value.red    { color: #dc2626; }
.kpi-value.amber  { color: #d97706; }
.kpi-value.blue   { color: #2563eb; }

/* Regime pill */
.pill {
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 4px 14px;
    border-radius: 999px;
    margin-top: 10px;
    text-transform: uppercase;
}
.pill-green  { background: #d1fae5; color: #065f46; }
.pill-red    { background: #fee2e2; color: #991b1b; }
.pill-blue   { background: #dbeafe; color: #1e40af; }
.pill-gray   { background: #f3f4f6; color: #374151; }

/* Allocation row */
.alloc-wrap { margin: 6px 0; }
.alloc-row  { display: flex; align-items: center; gap: 14px; margin: 10px 0; }
.alloc-ticker { font-family: 'JetBrains Mono', monospace; font-size: 13px;
                font-weight: 600; color: #374151; width: 44px; text-align: right; }
.alloc-bg   { flex: 1; height: 8px; background: #f3f4f6; border-radius: 4px; overflow: hidden; }
.alloc-fill { height: 100%; border-radius: 4px; }
.alloc-pct  { font-family: 'JetBrains Mono', monospace; font-size: 14px;
               font-weight: 600; color: #111827; width: 50px; }

/* Section header */
.sec-head {
    font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #9ca3af;
    border-bottom: 1px solid #f3f4f6;
    padding-bottom: 10px; margin: 28px 0 18px;
}

/* Kelly grid */
.kelly-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 4px; }
.kelly-item { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px 18px; }
.kelly-name { font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
               text-transform: uppercase; color: #9ca3af; margin-bottom: 6px; }
.kelly-val  { font-family: 'JetBrains Mono', monospace; font-size: 24px;
               font-weight: 600; color: #059669; }

/* Gate badges */
.gate-row { display: flex; gap: 16px; margin-top: 14px; }
.gate-item { text-align: center; }
.gate-label { font-size: 11px; font-weight: 600; color: #9ca3af; margin-bottom: 4px; }
.gate-val   { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 600; }
.gate-A { color: #dc2626; }
.gate-B { color: #059669; }
.gate-C { color: #2563eb; }

/* Rule card */
.rule-card {
    border: 1px solid #e5e7eb; border-left: 4px solid #6c5ce7;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
    background: #fafafa;
}
.rule-title { font-size: 13px; font-weight: 600; color: #6c5ce7; margin-bottom: 6px; }
.rule-body  { font-size: 15px; color: #1f2937; line-height: 1.6; }
.rule-meta  { font-family: 'JetBrains Mono', monospace; font-size: 11px;
               color: #9ca3af; margin-top: 10px; }

/* Prob bar */
.prob-row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
.prob-id   { font-family: 'JetBrains Mono', monospace; font-size: 12px;
              color: #9ca3af; width: 20px; }
.prob-name { font-size: 13px; font-weight: 500; color: #374151; width: 170px; }
.prob-bg   { flex: 1; height: 6px; background: #f3f4f6; border-radius: 3px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 3px; background: #059669; }
.prob-pct  { font-family: 'JetBrains Mono', monospace; font-size: 12px;
              color: #111827; width: 38px; text-align: right; }

/* Tabs */
div[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #111827 !important;
    border-bottom: 2px solid #059669 !important;
}

/* Page title */
.page-title {
    font-size: 28px; font-weight: 700; color: #059669;
    letter-spacing: -0.02em; display: inline;
}
.page-sub {
    font-size: 13px; font-weight: 500; color: #9ca3af;
    letter-spacing: 0.05em; margin-left: 14px;
}
.page-desc { font-size: 14px; color: #6b7280; margin-top: 4px; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_signal():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.LATEST_SIGNAL_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN, force_download=True)
        with open(p) as f: return json.load(f)
    except: return None

@st.cache_data(ttl=300)
def load_history():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.SIGNAL_HISTORY_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN, force_download=True)
        with open(p) as f: return json.load(f)
    except: return []

@st.cache_data(ttl=300)
def load_perf():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.PERFORMANCE_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN, force_download=True)
        with open(p) as f: return json.load(f)
    except: return None

@st.cache_data(ttl=300)
def load_rulebook():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.RULEBOOK_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN, force_download=True)
        with open(p) as f: return json.load(f)
    except: return None

@st.cache_data(ttl=300)
def load_regime_hist():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.REGIME_HISTORY_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN, force_download=True)
        return pd.read_csv(p, parse_dates=['date'])
    except: return None


# ── Helpers ────────────────────────────────────────────────────────────────────
ASSET_COLORS = {
    'TLT':'#2563eb','LQD':'#7c3aed','HYG':'#d97706',
    'VNQ':'#0891b2','GLD':'#ca8a04','SLV':'#6b7280','CASH':'#d1d5db',
}

def regime_pill(name):
    n = (name or '').lower()
    if any(k in n for k in ['crisis','stress','risk off']): cls = 'pill-red'
    elif any(k in n for k in ['expansion','growth','recovery']): cls = 'pill-green'
    elif 'flat' in n or 'late' in n: cls = 'pill-blue'
    else: cls = 'pill-gray'
    return f'<span class="pill {cls}">{name}</span>'

def pct(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return '—'
    sign = '+' if v >= 0 else ''
    return f"{sign}{v*100:.{d}f}%"

def flt(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)): return '—'
    return f"{v:.{d}f}"

def color_class(v, good_positive=True):
    if v is None or (isinstance(v, float) and np.isnan(v)): return ''
    if good_positive: return 'green' if v >= 0 else 'red'
    return 'red' if v >= 0 else 'green'


# ── Charts ─────────────────────────────────────────────────────────────────────
def equity_chart(history):
    scored = [s for s in history if s.get('scored')]
    if len(scored) < 2: return None
    dates = pd.to_datetime([s['date'] for s in scored])
    p_cum = (1 + np.array([s['portfolio_return'] for s in scored])).cumprod() * 100
    b_cum = (1 + np.array([s['benchmark_return'] for s in scored])).cumprod() * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=p_cum, name='REALM',
        line=dict(color='#059669', width=2.5),
        fill='tozeroy', fillcolor='rgba(5,150,105,0.06)'))
    fig.add_trace(go.Scatter(x=dates, y=b_cum, name='AGG',
        line=dict(color='#9ca3af', width=1.5, dash='dot')))
    fig.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        height=240, margin=dict(l=0,r=0,t=8,b=0),
        font=dict(family='Inter', color='#6b7280', size=11),
        legend=dict(orientation='h', y=1.12),
        xaxis=dict(gridcolor='#f3f4f6', showline=False),
        yaxis=dict(gridcolor='#f3f4f6', title='Base 100'),
    )
    return fig

def excess_chart(history):
    scored = [s for s in history if s.get('scored')][-30:]
    if not scored: return None
    dates = [s['date'][-5:] for s in scored]
    vals  = [s['excess_return']*100 for s in scored]
    colors = ['#059669' if v >= 0 else '#dc2626' for v in vals]
    fig = go.Figure(go.Bar(x=dates, y=vals, marker_color=colors, marker_line_width=0))
    fig.add_hline(y=0, line_color='#e5e7eb', line_width=1)
    fig.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        height=200, margin=dict(l=0,r=0,t=8,b=0),
        font=dict(family='Inter', color='#6b7280', size=11),
        xaxis=dict(gridcolor='#f3f4f6', tickangle=-45),
        yaxis=dict(gridcolor='#f3f4f6', ticksuffix='%'),
        showlegend=False,
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    signal  = load_signal()
    history = load_history()
    perf    = load_perf()
    rulebook_data = load_rulebook()
    regime_hist   = load_regime_hist()

    # Header
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px">
        <div>
            <span class="page-title">REALM</span>
            <span class="page-sub">P2 ETF ERL ENGINE</span>
            <div class="page-desc">Regime-Aware Experiential Asset Learning Machine</div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#9ca3af;margin-top:6px">{now}</div>
    </div>
    """, unsafe_allow_html=True)

    # Stale warning
    if signal:
        try:
            gen = datetime.fromisoformat(signal.get('generated_at',''))
            if (datetime.utcnow()-gen).total_seconds() > 86400*2:
                st.warning("⚠ Signal is more than 2 days old")
        except: pass
    else:
        st.error("No signal available yet.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Signal", "Regime", "Performance", "Rulebook"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — SIGNAL
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        rname = signal.get('regime_name','Unknown')
        kf    = signal.get('kelly_fraction') or 0
        cp    = signal.get('crisis_prob') or 0
        rs    = signal.get('rolling_sharpe') or 0

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Current Regime</div>
                <div class="kpi-value" style="font-size:22px;font-family:Inter">{rname}</div>
                {regime_pill(rname)}
            </div>""", unsafe_allow_html=True)
        with c2:
            kf_v = kf if not np.isnan(kf) else 0
            kf_c = 'green' if kf_v >= 0.2 else 'amber' if kf_v >= 0.1 else 'red'
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Kelly Fraction</div>
                <div class="kpi-value {kf_c}">{kf_v:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            cp_c = 'red' if cp > 0.4 else 'amber' if cp > 0.2 else 'green'
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Crisis Probability</div>
                <div class="kpi-value {cp_c}">{cp:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            rs_c = 'green' if rs >= 1 else 'amber' if rs >= 0 else 'red'
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Rolling Sharpe</div>
                <div class="kpi-value {rs_c}">{rs:.2f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1.3, 1])

        with col_left:
            st.markdown('<div class="sec-head">Allocation</div>', unsafe_allow_html=True)
            alloc = signal.get('allocation', {})
            if alloc:
                rows = ''
                for asset, w in sorted(alloc.items(), key=lambda x: -x[1]):
                    color = ASSET_COLORS.get(asset, '#059669')
                    rows += f"""<div class="alloc-row">
                        <span class="alloc-ticker">{asset}</span>
                        <div class="alloc-bg"><div class="alloc-fill"
                            style="width:{w*100:.1f}%;background:{color}"></div></div>
                        <span class="alloc-pct">{w*100:.1f}%</span>
                    </div>"""
                st.markdown(f'<div class="alloc-wrap">{rows}</div>', unsafe_allow_html=True)

            turn = signal.get('turnover', 0)
            top  = signal.get('top_asset', '—')
            date = signal.get('date', '—')
            st.markdown(f"""<div style="font-size:12px;color:#9ca3af;margin-top:12px;
                font-family:'JetBrains Mono',monospace">
                TURNOVER {turn:.1%} &nbsp;·&nbsp; TOP {top} &nbsp;·&nbsp; DATE {date}
            </div>""", unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="sec-head">Kelly Sizing</div>', unsafe_allow_html=True)
            rs_sc  = signal.get('kelly_regime_scalar', 0) or 0
            ag_sc  = signal.get('kelly_agreement_scalar', 0) or 0
            sh_sc  = signal.get('kelly_sharpe_scalar', 0) or 0
            st.markdown(f"""<div class="kelly-grid">
                <div class="kelly-item">
                    <div class="kelly-name">Base</div>
                    <div class="kelly-val">{cfg.KELLY_BASE_FRACTION:.2f}</div>
                </div>
                <div class="kelly-item">
                    <div class="kelly-name">Regime ×</div>
                    <div class="kelly-val">{rs_sc:.2f}</div>
                </div>
                <div class="kelly-item">
                    <div class="kelly-name">Agreement ×</div>
                    <div class="kelly-val">{ag_sc:.2f}</div>
                </div>
                <div class="kelly-item">
                    <div class="kelly-name">Sharpe ×</div>
                    <div class="kelly-val">{sh_sc:.2f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            gA = signal.get('gate_A', 0) or 0
            gB = signal.get('gate_B', 0) or 0
            gC = signal.get('gate_C', 0) or 0
            st.markdown(f"""
            <div class="sec-head" style="margin-top:24px">Ensemble Gates</div>
            <div class="gate-row">
                <div class="gate-item">
                    <div class="gate-label">A · Crisis</div>
                    <div class="gate-val gate-A">{gA:.2f}</div>
                </div>
                <div class="gate-item">
                    <div class="gate-label">B · Expansion</div>
                    <div class="gate-val gate-B">{gB:.2f}</div>
                </div>
                <div class="gate-item">
                    <div class="gate-label">C · Full</div>
                    <div class="gate-val gate-C">{gC:.2f}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        rules = signal.get('active_rule_summary', [])
        if rules:
            st.markdown('<div class="sec-head">Active Rules</div>', unsafe_allow_html=True)
            for r in rules:
                st.markdown(f'<div class="rule-card"><div class="rule-body">{r}</div></div>',
                            unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — REGIME
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        c1, c2 = st.columns([1, 1.8])
        with c1:
            st.markdown('<div class="sec-head">Regime Probabilities</div>', unsafe_allow_html=True)
            hmm = signal.get('hmm_probs', {})
            for k in range(cfg.HMM_N_STATES):
                p   = float(hmm.get(str(k), 0))
                nm  = cfg.REGIME_NAMES.get(k, f'Regime {k}')
                w   = int(p * 100)
                st.markdown(f"""<div class="prob-row">
                    <span class="prob-id">{k}</span>
                    <span class="prob-name">{nm}</span>
                    <div class="prob-bg"><div class="prob-fill" style="width:{w}%"></div></div>
                    <span class="prob-pct">{p:.0%}</span>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="sec-head">Regime Timeline (Last Year)</div>',
                        unsafe_allow_html=True)
            if regime_hist is not None and len(regime_hist) > 0:
                recent  = regime_hist.tail(252)
                palette = ['#059669','#7c3aed','#d97706','#0891b2',
                           '#dc2626','#db2777','#6b7280','#ca8a04']
                fig = go.Figure()
                for k in range(cfg.HMM_N_STATES):
                    mask = recent['regime'] == k
                    if mask.sum() == 0: continue
                    fig.add_trace(go.Scatter(
                        x=recent[mask]['date'], y=[1]*mask.sum(),
                        mode='markers',
                        marker=dict(symbol='square', size=8, color=palette[k]),
                        name=cfg.REGIME_NAMES.get(k, f'R{k}'),
                    ))
                fig.update_layout(
                    paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                    height=120, margin=dict(l=0,r=0,t=8,b=0),
                    font=dict(family='Inter', color='#6b7280', size=11),
                    xaxis=dict(gridcolor='#f3f4f6'),
                    yaxis=dict(visible=False),
                    legend=dict(orientation='h', y=1.5, font=dict(size=11)),
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={'displayModeBar': False})
            else:
                st.info("Regime history not available yet.")

            st.markdown('<div class="sec-head" style="margin-top:20px">Transition Entropy</div>',
                        unsafe_allow_html=True)
            ent     = signal.get('transition_entropy', 0) or 0
            max_e   = np.log(cfg.HMM_N_STATES)
            e_pct   = ent / max_e if max_e > 0 else 0
            e_c     = 'red' if e_pct > 0.7 else 'amber' if e_pct > 0.4 else 'green'
            st.markdown(f"""<div class="kpi-card" style="margin-top:0">
                <div class="kpi-label">Entropy (0 = certain · 1 = max uncertainty)</div>
                <div class="kpi-value {e_c}">{ent:.3f}
                    <span style="font-size:16px;color:#9ca3af;font-family:Inter;font-weight:400">
                        / {max_e:.2f} &nbsp;({e_pct:.0%})
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        if not perf or perf.get('insufficient_data'):
            st.info("Not enough scored signals yet — performance metrics appear after the first scored trading day.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            metrics = [
                (c1, "Ann. Excess Return", perf.get('ann_excess_return'), True),
                (c2, "Excess Sharpe",      perf.get('excess_sharpe'),     None),
                (c3, "Win Rate vs AGG",    perf.get('win_rate_vs_bench'), True),
                (c4, "Max Drawdown",       perf.get('max_drawdown'),      False),
            ]
            for col, label, val, gp in metrics:
                with col:
                    if label == "Excess Sharpe":
                        disp = flt(val)
                        c = 'green' if val and val >= 0.5 else 'amber' if val and val >= 0 else 'red'
                    elif label == "Max Drawdown":
                        disp = pct(val)
                        c = 'red' if val and val < -0.05 else 'amber' if val and val < -0.02 else 'green'
                    else:
                        disp = pct(val)
                        c = color_class(val, gp)
                    st.markdown(f"""<div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value {c}">{disp}</div>
                    </div>""", unsafe_allow_html=True)

        ec = equity_chart(history)
        if ec:
            st.markdown('<div class="sec-head">Equity Curve (Base 100)</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(ec, use_container_width=True, config={'displayModeBar': False})

        eb = excess_chart(history)
        if eb:
            st.markdown('<div class="sec-head">Daily Excess Return vs AGG — Last 30 Days</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(eb, use_container_width=True, config={'displayModeBar': False})

        if perf and not perf.get('insufficient_data'):
            rp = perf.get('regime_performance', {})
            if rp:
                st.markdown('<div class="sec-head">Performance by Regime</div>',
                            unsafe_allow_html=True)
                rows = [{'Regime': k, 'Days': v['n_days'],
                         'Mean Daily Excess': f"{v['mean_excess']*100:+.3f}%",
                         'Win Rate': f"{v['win_rate']:.0%}"}
                        for k, v in rp.items()]
                st.dataframe(pd.DataFrame(rows).sort_values('Days', ascending=False),
                             hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — RULEBOOK
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        if not rulebook_data:
            st.info("No rulebook yet — run ERL training first.")
            return

        rules = rulebook_data.get('rules', [])
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Rules Stored</div>
                <div class="kpi-value blue">{len(rules)}/{cfg.ERL_MAX_RULES}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            stored   = rulebook_data.get('total_stored', 0)
            rejected = rulebook_data.get('total_rejected', 0)
            total    = stored + rejected
            gpr = stored/total if total > 0 else 0
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Gate Pass Rate</div>
                <div class="kpi-value {'green' if gpr>=0.5 else 'amber'}">{gpr:.0%}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            upd = rulebook_data.get('last_updated','')[:10]
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Last Updated</div>
                <div class="kpi-value" style="font-size:22px">{upd}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            gn = sum(1 for r in rules if r.get('source')=='gemini')
            st.markdown(f"""<div class="kpi-card">
                <div class="kpi-label">Gemini Rules</div>
                <div class="kpi-value">{gn}/{len(rules)}</div>
            </div>""", unsafe_allow_html=True)

        all_regimes = sorted(set(r.get('regime_name','Unknown') for r in rules))
        sel = st.selectbox("Filter by regime", ['All'] + all_regimes,
                           label_visibility='collapsed')
        filtered = rules if sel == 'All' else [r for r in rules if r.get('regime_name')==sel]
        filtered = sorted(filtered, key=lambda r: -r.get('confidence', 0))

        st.markdown(f"""<div style="font-size:12px;color:#9ca3af;margin:12px 0 20px;
            font-family:'JetBrains Mono',monospace">
            SHOWING {len(filtered)} RULE(S)
        </div>""", unsafe_allow_html=True)

        for r in filtered:
            conf  = r.get('confidence', 0)
            impr  = r.get('improvement', 0) * 10000  # to bp
            src   = r.get('source','?').upper()
            rtype = r.get('rule_type','?').upper()
            asset = r.get('primary_asset','?')
            rname = r.get('regime_name','?')
            action= r.get('action', r.get('rationale',''))
            stored= r.get('stored_at','')[:10]
            ep    = r.get('episode_id','?')
            bc    = '#059669' if conf >= 0.7 else '#d97706' if conf >= 0.5 else '#9ca3af'

            st.markdown(f"""<div class="rule-card" style="border-left-color:{bc}">
                <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                    <span class="rule-title">{rtype} · {asset}</span>
                    <span style="font-size:12px;color:#9ca3af;font-weight:500">{rname}</span>
                </div>
                <div class="rule-body">{action}</div>
                <div class="rule-meta">
                    CONF {conf:.2f} &nbsp;·&nbsp;
                    IMPROVEMENT +{impr:.1f}bp &nbsp;·&nbsp;
                    SOURCE {src} &nbsp;·&nbsp;
                    EPISODE {ep} &nbsp;·&nbsp;
                    STORED {stored}
                </div>
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# streamlit_app.py — REALM Dashboard for P2-ETF-ERL-ENGINE
# Displays live signal, regime state, portfolio allocation,
# performance history, and rulebook.
#
# Run locally:  streamlit run streamlit_app.py
# Deploy:       Streamlit Community Cloud → connect to GitHub repo

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "REALM · P2 ETF Engine",
    page_icon  = "⬡",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0b0e14;
    --surface:  #111620;
    --border:   #1e2535;
    --accent:   #4af0b0;
    --accent2:  #7c6af5;
    --warn:     #f5a623;
    --danger:   #f55c47;
    --text:     #d4dde8;
    --muted:    #5a6a80;
    --mono:     'Space Mono', monospace;
    --sans:     'DM Sans', sans-serif;
}

html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

.stApp { background: var(--bg) !important; }

/* Metric cards */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
}
.metric-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: var(--mono);
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.metric-value.positive { color: var(--accent); }
.metric-value.negative { color: var(--danger); }
.metric-value.warning  { color: var(--warn); }

/* Regime badge */
.regime-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    padding: 4px 12px;
    border-radius: 3px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.regime-crisis    { background: rgba(245,92,71,0.15);  color: #f55c47; border: 1px solid rgba(245,92,71,0.3); }
.regime-expansion { background: rgba(74,240,176,0.1);  color: #4af0b0; border: 1px solid rgba(74,240,176,0.3); }
.regime-neutral   { background: rgba(124,106,245,0.1); color: #7c6af5; border: 1px solid rgba(124,106,245,0.3); }

/* Section headers */
.section-title {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* Allocation bar */
.alloc-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 12px;
}
.alloc-label {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted);
    width: 46px;
    text-align: right;
}
.alloc-bar-bg {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.alloc-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--accent);
    transition: width 0.6s ease;
}
.alloc-pct {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    width: 44px;
}

/* Rule card */
.rule-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 13px;
}
.rule-meta {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 6px;
}

/* Kelly bars */
.kelly-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 8px;
}
.kelly-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
}
.kelly-name  { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.1em; }
.kelly-score { font-family: var(--mono); font-size: 18px; font-weight: 700; color: var(--accent); }

/* Stale warning */
.stale-banner {
    background: rgba(245,166,35,0.1);
    border: 1px solid rgba(245,166,35,0.3);
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 13px;
    color: var(--warn);
    margin-bottom: 16px;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.stDeployButton { display: none; }
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface) !important;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    border-radius: 5px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--border) !important;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data Loaders (cached) ──────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_latest_signal():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.LATEST_SIGNAL_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN,
                            force_download=True)
        with open(p) as f:
            return json.load(f)
    except:
        return None

@st.cache_data(ttl=300)
def load_signal_history():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.SIGNAL_HISTORY_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN,
                            force_download=True)
        with open(p) as f:
            return json.load(f)
    except:
        return []

@st.cache_data(ttl=300)
def load_performance():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.PERFORMANCE_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN,
                            force_download=True)
        with open(p) as f:
            return json.load(f)
    except:
        return None

@st.cache_data(ttl=300)
def load_rulebook():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.RULEBOOK_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN,
                            force_download=True)
        with open(p) as f:
            return json.load(f)
    except:
        return None

@st.cache_data(ttl=300)
def load_regime_history():
    try:
        p = hf_hub_download(cfg.HF_RESULTS_REPO, cfg.REGIME_HISTORY_PATH,
                            repo_type="dataset", token=cfg.HF_TOKEN,
                            force_download=True)
        return pd.read_csv(p, parse_dates=['date'])
    except:
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def regime_class(regime_name):
    n = regime_name.lower() if regime_name else ''
    if any(k in n for k in ['crisis', 'risk off', 'stress']):
        return 'regime-crisis'
    elif any(k in n for k in ['expansion', 'growth', 'recovery']):
        return 'regime-expansion'
    return 'regime-neutral'

def fmt_pct(v, decimals=1):
    if v is None: return '—'
    sign = '+' if v >= 0 else ''
    return f"{sign}{v*100:.{decimals}f}%"

def fmt_float(v, decimals=3):
    if v is None: return '—'
    return f"{v:.{decimals}f}"

def signal_is_stale(signal):
    if not signal: return True
    try:
        gen = datetime.fromisoformat(signal.get('generated_at', ''))
        return (datetime.utcnow() - gen).total_seconds() > 86400 * 2
    except:
        return True

ASSET_COLORS = {
    'TLT':'#4af0b0','LQD':'#7c6af5','HYG':'#f5a623',
    'VNQ':'#4ab8f0','GLD':'#f5d84a','SLV':'#a0a8b8','CASH':'#2a3448',
}


# ── Chart Builders ─────────────────────────────────────────────────────────────

def chart_equity_curve(history):
    scored = [s for s in history if s.get('scored')]
    if len(scored) < 2:
        return None
    dates  = pd.to_datetime([s['date'] for s in scored])
    p_ret  = np.array([s['portfolio_return']  for s in scored])
    b_ret  = np.array([s['benchmark_return']  for s in scored])
    p_cum  = (1 + p_ret).cumprod() * 100
    b_cum  = (1 + b_ret).cumprod() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=p_cum, name='REALM',
        line=dict(color='#4af0b0', width=2),
        fill='tozeroy', fillcolor='rgba(74,240,176,0.05)',
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=b_cum, name='AGG',
        line=dict(color='#5a6a80', width=1.5, dash='dot'),
    ))
    fig.update_layout(
        paper_bgcolor='#0b0e14', plot_bgcolor='#0b0e14',
        font=dict(family='Space Mono', color='#5a6a80', size=10),
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation='h', y=1.08, font=dict(size=10)),
        xaxis=dict(gridcolor='#1e2535', tickfont=dict(size=9)),
        yaxis=dict(gridcolor='#1e2535', tickfont=dict(size=9),
                   ticksuffix='', title='Base 100'),
        height=220,
    )
    return fig


def chart_excess_bars(history):
    scored = [s for s in history if s.get('scored')][-30:]
    if not scored: return None
    dates  = [s['date'][-5:] for s in scored]  # MM-DD
    excess = [s['excess_return'] * 100 for s in scored]
    colors = ['#4af0b0' if e >= 0 else '#f55c47' for e in excess]

    fig = go.Figure(go.Bar(x=dates, y=excess, marker_color=colors,
                           marker_line_width=0))
    fig.update_layout(
        paper_bgcolor='#0b0e14', plot_bgcolor='#0b0e14',
        font=dict(family='Space Mono', color='#5a6a80', size=9),
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(gridcolor='#1e2535', tickfont=dict(size=8),
                   tickangle=-45),
        yaxis=dict(gridcolor='#1e2535', tickfont=dict(size=9),
                   ticksuffix='%', title='Daily excess (%)'),
        height=180,
        showlegend=False,
    )
    fig.add_hline(y=0, line_color='#1e2535', line_width=1)
    return fig


def chart_regime_donut(hmm_probs):
    labels = [cfg.REGIME_NAMES.get(k, f'R{k}') for k in range(cfg.HMM_N_STATES)]
    values = [hmm_probs.get(str(k), 0) for k in range(cfg.HMM_N_STATES)]
    palette = ['#4af0b0','#7c6af5','#f5a623','#4ab8f0',
               '#f5d84a','#f55c47','#e040fb','#a0a8b8']
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.65, marker_colors=palette,
        textinfo='none',
        hovertemplate='%{label}: %{percent}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor='transparent', plot_bgcolor='transparent',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        height=140,
    )
    return fig


def chart_regime_timeline(regime_history):
    if regime_history is None or len(regime_history) == 0:
        return None
    recent = regime_history.tail(252)   # last year
    palette = {k: v for k, v in zip(
        range(cfg.HMM_N_STATES),
        ['#4af0b0','#7c6af5','#f5a623','#4ab8f0',
         '#f55c47','#e040fb','#a0a8b8','#f5d84a']
    )}
    fig = go.Figure()
    for k in range(cfg.HMM_N_STATES):
        mask = recent['regime'] == k
        if mask.sum() == 0: continue
        fig.add_trace(go.Scatter(
            x=recent[mask]['date'], y=[1]*mask.sum(),
            mode='markers', marker=dict(
                symbol='square', size=6,
                color=palette[k],
            ),
            name=cfg.REGIME_NAMES.get(k, f'R{k}'),
            hovertemplate=f"{cfg.REGIME_NAMES.get(k,'R'+str(k))}<br>%{{x}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor='#0b0e14', plot_bgcolor='#0b0e14',
        font=dict(family='Space Mono', color='#5a6a80', size=9),
        margin=dict(l=0, r=0, t=8, b=0),
        height=100,
        xaxis=dict(gridcolor='#1e2535', tickfont=dict(size=8)),
        yaxis=dict(visible=False),
        legend=dict(orientation='h', y=1.5, font=dict(size=8)),
        showlegend=True,
    )
    return fig


def allocation_bars_html(allocation):
    if not allocation: return '<p style="color:#5a6a80">No allocation data</p>'
    rows = ''
    for asset, w in sorted(allocation.items(), key=lambda x: -x[1]):
        pct    = w * 100
        color  = ASSET_COLORS.get(asset, '#4af0b0')
        rows += f"""
        <div class="alloc-row">
            <span class="alloc-label">{asset}</span>
            <div class="alloc-bar-bg">
                <div class="alloc-bar-fill"
                     style="width:{pct:.1f}%;background:{color}"></div>
            </div>
            <span class="alloc-pct">{pct:.1f}%</span>
        </div>"""
    return rows


# ── App ────────────────────────────────────────────────────────────────────────

def main():
    signal  = load_latest_signal()
    history = load_signal_history()
    perf    = load_performance()
    rulebook_data = load_rulebook()
    regime_hist   = load_regime_history()

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.markdown("""
        <div style="display:flex;align-items:baseline;gap:14px;margin-bottom:4px">
            <span style="font-family:'Space Mono';font-size:22px;
                         font-weight:700;color:#4af0b0;letter-spacing:0.05em">
                REALM
            </span>
            <span style="font-family:'Space Mono';font-size:11px;
                         color:#5a6a80;letter-spacing:0.15em">
                P2 ETF ERL ENGINE
            </span>
        </div>
        <div style="font-family:'DM Sans';font-size:13px;color:#5a6a80">
            Regime-Aware Experiential Asset Learning Machine
        </div>
        """, unsafe_allow_html=True)
    with col_date:
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        st.markdown(f"""
        <div style="text-align:right;margin-top:8px">
            <span style="font-family:'Space Mono';font-size:10px;
                         color:#5a6a80">{now}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    if signal_is_stale(signal):
        st.markdown("""
        <div class="stale-banner">
            ⚠ Signal is more than 2 days old — model may not have run recently
        </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_signal, tab_regime, tab_perf, tab_rules = st.tabs([
        "SIGNAL", "REGIME", "PERFORMANCE", "RULEBOOK"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — SIGNAL
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_signal:
        if not signal:
            st.warning("No signal available yet. Ensure predict.py has run.")
            return

        regime_name = signal.get('regime_name', 'Unknown')
        rc = regime_class(regime_name)

        # Top row: regime + kelly + basis
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Regime</div>
                <div class="metric-value">{regime_name}</div>
                <div style="margin-top:8px">
                    <span class="regime-badge {rc}">{rc.replace('regime-','')}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            kf = signal.get('kelly_fraction', 0)
            cls = 'positive' if kf >= 0.25 else 'warning' if kf >= 0.15 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Kelly Fraction</div>
                <div class="metric-value {cls}">{kf:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            cp = signal.get('crisis_prob', 0)
            cls = 'negative' if cp > 0.4 else 'warning' if cp > 0.2 else 'positive'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Crisis Prob</div>
                <div class="metric-value {cls}">{cp:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            rs = signal.get('rolling_sharpe', 0)
            cls = 'positive' if rs >= 1 else 'warning' if rs >= 0 else 'negative'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rolling Sharpe</div>
                <div class="metric-value {cls}">{rs:.2f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

        # Allocation + Kelly breakdown
        col_alloc, col_kelly = st.columns([1.2, 1])

        with col_alloc:
            st.markdown('<div class="section-title">Allocation</div>',
                        unsafe_allow_html=True)
            allocation = signal.get('allocation', {})
            st.markdown(allocation_bars_html(allocation), unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Space Mono';font-size:10px;
                        color:#5a6a80;margin-top:8px">
                TURNOVER {signal.get('turnover',0):.1%} &nbsp;·&nbsp;
                TOP ASSET {signal.get('top_asset','—')} &nbsp;·&nbsp;
                SIGNAL DATE {signal.get('date','—')}
            </div>""", unsafe_allow_html=True)

        with col_kelly:
            st.markdown('<div class="section-title">Kelly Components</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="kelly-grid">
                <div class="kelly-item">
                    <div class="kelly-name">BASE</div>
                    <div class="kelly-score">{cfg.KELLY_BASE_FRACTION:.2f}</div>
                </div>
                <div class="kelly-item">
                    <div class="kelly-name">REGIME ×</div>
                    <div class="kelly-score">{signal.get('kelly_regime_scalar',0):.2f}</div>
                </div>
                <div class="kelly-item">
                    <div class="kelly-name">AGREEMENT ×</div>
                    <div class="kelly-score">{signal.get('kelly_agreement_scalar',0):.2f}</div>
                </div>
                <div class="kelly-item">
                    <div class="kelly-name">SHARPE ×</div>
                    <div class="kelly-score">{signal.get('kelly_sharpe_scalar',0):.2f}</div>
                </div>
            </div>
            <div style="margin-top:14px;font-family:'Space Mono';
                        font-size:10px;color:#5a6a80">ENSEMBLE GATES</div>
            <div style="display:flex;gap:12px;margin-top:8px">
                <div style="font-family:'Space Mono';font-size:12px">
                    <span style="color:#f55c47">A(crisis)</span>
                    <span style="color:#d4dde8;margin-left:4px">
                        {signal.get('gate_A',0):.2f}
                    </span>
                </div>
                <div style="font-family:'Space Mono';font-size:12px">
                    <span style="color:#4af0b0">B(expand)</span>
                    <span style="color:#d4dde8;margin-left:4px">
                        {signal.get('gate_B',0):.2f}
                    </span>
                </div>
                <div style="font-family:'Space Mono';font-size:12px">
                    <span style="color:#7c6af5">C(full)</span>
                    <span style="color:#d4dde8;margin-left:4px">
                        {signal.get('gate_C',0):.2f}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Active rules
        rules = signal.get('active_rule_summary', [])
        if rules:
            st.markdown('<div class="section-title">Active Rules</div>',
                        unsafe_allow_html=True)
            for r in rules:
                st.markdown(f'<div class="rule-card">{r}</div>',
                            unsafe_allow_html=True)

        # Basis note
        basis = signal.get('basis', 'live')
        st.markdown(f"""
        <div style="font-family:'Space Mono';font-size:9px;
                    color:#2a3448;margin-top:16px">
            SIGNAL BASIS: {basis.upper()} &nbsp;·&nbsp;
            GENERATED {signal.get('generated_at','')[:16]} UTC
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — REGIME
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_regime:
        if not signal:
            st.info("No signal available.")
            return

        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.markdown('<div class="section-title">Regime Probabilities</div>',
                        unsafe_allow_html=True)
            hmm_probs = signal.get('hmm_probs', {})
            donut = chart_regime_donut(hmm_probs)
            if donut:
                regime_name = signal.get('regime_name', '')
                st.markdown(f"""
                <div style="position:relative;text-align:center">
                    <div style="font-family:'Space Mono';font-size:11px;
                                color:#4af0b0;letter-spacing:0.1em;
                                margin-bottom:-4px">{regime_name}</div>
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(donut, use_container_width=True,
                                config={'displayModeBar': False})

            # Prob table
            for k in range(cfg.HMM_N_STATES):
                p    = float(hmm_probs.get(str(k), 0))
                name = cfg.REGIME_NAMES.get(k, f'R{k}')
                bar  = '█' * int(p * 20)
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;
                            margin-bottom:4px;font-family:'Space Mono';
                            font-size:10px">
                    <span style="color:#5a6a80;width:24px">{k}</span>
                    <span style="color:#d4dde8;width:160px">{name}</span>
                    <span style="color:#4af0b0;flex:1">{bar}</span>
                    <span style="color:#d4dde8;width:36px;text-align:right">
                        {p:.0%}
                    </span>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-title">Regime Timeline (Last Year)</div>',
                        unsafe_allow_html=True)
            rtl = chart_regime_timeline(regime_hist)
            if rtl:
                st.plotly_chart(rtl, use_container_width=True,
                                config={'displayModeBar': False})
            else:
                st.info("Regime history not yet available.")

            st.markdown('<div class="section-title" style="margin-top:20px">'
                        'Transition Entropy</div>', unsafe_allow_html=True)
            entropy = signal.get('transition_entropy', 0)
            max_e   = np.log(cfg.HMM_N_STATES)
            e_pct   = entropy / max_e
            cls     = 'negative' if e_pct > 0.7 else 'warning' if e_pct > 0.4 else 'positive'
            st.markdown(f"""
            <div class="metric-card" style="margin-top:0">
                <div class="metric-label">Entropy (0 = certain, 1 = max uncertainty)</div>
                <div class="metric-value {cls}">{entropy:.3f}
                    <span style="font-size:13px;color:#5a6a80">
                        / {max_e:.3f} &nbsp;({e_pct:.0%})
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_perf:
        if not perf or perf.get('insufficient_data'):
            st.info("Not enough scored signals yet for performance metrics.")
        else:
            # Top metrics
            c1, c2, c3, c4 = st.columns(4)
            mets = [
                (c1, "Ann. Excess Return", perf.get('ann_excess_return'),
                 True, 2),
                (c2, "Excess Sharpe",      perf.get('excess_sharpe'),
                 None, 3),
                (c3, "Win Rate vs AGG",    perf.get('win_rate_vs_bench'),
                 True, 1),
                (c4, "Max Drawdown",       perf.get('max_drawdown'),
                 False, 2),
            ]
            for col, label, val, is_pct_exc, dec in mets:
                with col:
                    if val is None:
                        disp, cls = '—', ''
                    elif label == "Max Drawdown":
                        disp = fmt_pct(val, dec)
                        cls  = 'negative' if val < -0.05 else 'warning' \
                               if val < -0.02 else 'positive'
                    elif label == "Excess Sharpe":
                        disp = fmt_float(val, dec)
                        cls  = 'positive' if val >= 0.5 else 'warning' \
                               if val >= 0 else 'negative'
                    elif label == "Win Rate vs AGG":
                        disp = fmt_pct(val, dec)
                        cls  = 'positive' if val >= 0.55 else 'warning' \
                               if val >= 0.5 else 'negative'
                    else:
                        disp = fmt_pct(val, dec)
                        cls  = 'positive' if val >= 0 else 'negative'
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value {cls}">{disp}</div>
                    </div>""", unsafe_allow_html=True)

        # Equity curve
        ec = chart_equity_curve(history)
        if ec:
            st.markdown('<div class="section-title">Equity Curve (Base 100)</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(ec, use_container_width=True,
                            config={'displayModeBar': False})

        # Daily excess bars
        eb = chart_excess_bars(history)
        if eb:
            st.markdown('<div class="section-title">Daily Excess Return vs AGG (Last 30 Days)</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(eb, use_container_width=True,
                            config={'displayModeBar': False})

        # Rolling 20d metrics
        if perf and not perf.get('insufficient_data'):
            st.markdown('<div class="section-title">Rolling 20-Day</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                v = perf.get('rolling_20d_return')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">20d Ann. Return</div>
                    <div class="metric-value {'positive' if v and v>=0 else 'negative'}">
                        {fmt_pct(v)}
                    </div>
                </div>""", unsafe_allow_html=True)
            with c2:
                v = perf.get('rolling_20d_excess')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">20d Ann. Excess</div>
                    <div class="metric-value {'positive' if v and v>=0 else 'negative'}">
                        {fmt_pct(v)}
                    </div>
                </div>""", unsafe_allow_html=True)
            with c3:
                v = perf.get('rolling_20d_sharpe')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">20d Sharpe</div>
                    <div class="metric-value {'positive' if v and v>=0.5 else 'warning' if v and v>=0 else 'negative'}">
                        {fmt_float(v)}
                    </div>
                </div>""", unsafe_allow_html=True)

            # Regime performance table
            regime_perf = perf.get('regime_performance', {})
            if regime_perf:
                st.markdown('<div class="section-title">Performance by Regime</div>',
                            unsafe_allow_html=True)
                rows = []
                for rname, rdata in regime_perf.items():
                    rows.append({
                        'Regime':       rname,
                        'Days':         rdata['n_days'],
                        'Mean Excess':  f"{rdata['mean_excess']*100:+.2f}%",
                        'Win Rate':     f"{rdata['win_rate']:.0%}",
                    })
                df = pd.DataFrame(rows).sort_values('Days', ascending=False)
                st.dataframe(df, hide_index=True, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — RULEBOOK
    # ═══════════════════════════════════════════════════════════════════════════
    with tab_rules:
        if not rulebook_data:
            st.info("No rulebook found yet — run the ERL training loop first.")
            return

        rules = rulebook_data.get('rules', [])
        meta  = {k: v for k, v in rulebook_data.items() if k != 'rules'}

        # Meta stats
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rules Stored</div>
                <div class="metric-value">{len(rules)}/{cfg.ERL_MAX_RULES}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            stored   = rulebook_data.get('total_stored', 0)
            rejected = rulebook_data.get('total_rejected', 0)
            total    = stored + rejected
            gate_pct = stored / total if total > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Gate Pass Rate</div>
                <div class="metric-value">{gate_pct:.0%}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            last_upd = rulebook_data.get('last_updated', '')[:10]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Last Updated</div>
                <div class="metric-value" style="font-size:18px">{last_upd}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            gemini_n = sum(1 for r in rules if r.get('source') == 'gemini')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Gemini Rules</div>
                <div class="metric-value">{gemini_n}/{len(rules)}</div>
            </div>""", unsafe_allow_html=True)

        # Filter by regime
        all_regimes = sorted(set(
            r.get('regime_name', 'Unknown') for r in rules
        ))
        sel_regime = st.selectbox(
            "Filter by regime",
            ['All'] + all_regimes,
            label_visibility='collapsed',
        )

        filtered = rules if sel_regime == 'All' else [
            r for r in rules if r.get('regime_name') == sel_regime
        ]
        filtered = sorted(filtered, key=lambda r: -r.get('confidence', 0))

        st.markdown(f"""
        <div style="font-family:'Space Mono';font-size:10px;
                    color:#5a6a80;margin:12px 0 16px">
            SHOWING {len(filtered)} RULE(S)
        </div>""", unsafe_allow_html=True)

        for r in filtered:
            conf     = r.get('confidence', 0)
            impr     = r.get('improvement', 0)
            source   = r.get('source', '?')
            rtype    = r.get('rule_type', '?').upper()
            asset    = r.get('primary_asset', '?')
            regime   = r.get('regime_name', '?')
            action   = r.get('action', r.get('rationale', ''))
            stored   = r.get('stored_at', '')[:10]
            ep       = r.get('episode_id', '?')

            border_c = '#4af0b0' if conf >= 0.7 else \
                       '#f5a623' if conf >= 0.5 else '#5a6a80'

            st.markdown(f"""
            <div class="rule-card" style="border-left-color:{border_c}">
                <div style="display:flex;justify-content:space-between;
                            align-items:flex-start;margin-bottom:6px">
                    <span style="font-family:'Space Mono';font-size:11px;
                                 color:{border_c}">{rtype} {asset}</span>
                    <span style="font-family:'Space Mono';font-size:10px;
                                 color:#5a6a80">{regime}</span>
                </div>
                <div style="font-size:13px;color:#d4dde8;line-height:1.5">
                    {action}
                </div>
                <div class="rule-meta">
                    CONF {conf:.2f} &nbsp;·&nbsp;
                    IMPR {impr*100:+.1f}bp &nbsp;·&nbsp;
                    SOURCE {source.upper()} &nbsp;·&nbsp;
                    EP {ep} &nbsp;·&nbsp;
                    STORED {stored}
                </div>
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    background: #ffffff !important;
    color: #111111 !important;
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding: 2.5rem 3rem !important; max-width: 1300px; }

/* Cards */
.card {
    background: #ffffff;
    border: 1.5px solid #e0e0e0;
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 16px;
}
.card-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #555555;
    margin-bottom: 10px;
}
.card-value {
    font-size: 36px;
    font-weight: 700;
    color: #111111;
    line-height: 1.1;
}
.card-value.green { color: #15803d; }
.card-value.red   { color: #b91c1c; }
.card-value.amber { color: #b45309; }
.card-value.blue  { color: #1d4ed8; }

/* Regime pill */
.pill {
    display: inline-block;
    font-size: 13px;
    font-weight: 600;
    padding: 5px 16px;
    border-radius: 999px;
    margin-top: 12px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.pill-green { background: #dcfce7; color: #14532d; }
.pill-red   { background: #fee2e2; color: #7f1d1d; }
.pill-blue  { background: #dbeafe; color: #1e3a8a; }
.pill-gray  { background: #f3f4f6; color: #374151; }

/* Section header */
.sec-head {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888888;
    border-bottom: 2px solid #f0f0f0;
    padding-bottom: 10px;
    margin: 32px 0 20px;
}

/* Allocation bars */
.alloc-row { display: flex; align-items: center; gap: 16px; margin: 12px 0; }
.alloc-ticker {
    font-size: 15px; font-weight: 700; color: #111111;
    width: 48px; text-align: right;
}
.alloc-bg { flex: 1; height: 10px; background: #f0f0f0; border-radius: 5px; overflow: hidden; }
.alloc-fill { height: 100%; border-radius: 5px; }
.alloc-pct { font-size: 16px; font-weight: 700; color: #111111; width: 56px; }

/* Kelly grid */
.kelly-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.kelly-item {
    background: #f9f9f9; border: 1.5px solid #e0e0e0;
    border-radius: 10px; padding: 18px 22px;
}
.kelly-name { font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
               text-transform: uppercase; color: #777777; margin-bottom: 8px; }
.kelly-val  { font-size: 28px; font-weight: 700; color: #15803d; }

/* Gate row */
.gate-row { display: flex; gap: 32px; margin-top: 16px; }
.gate-item { text-align: center; }
.gate-label { font-size: 13px; font-weight: 600; color: #777777; margin-bottom: 6px; }
.gate-val   { font-size: 28px; font-weight: 700; }
.gate-A { color: #b91c1c; }
.gate-B { color: #15803d; }
.gate-C { color: #1d4ed8; }

/* Rule card */
.rule-card {
    border: 1.5px solid #e0e0e0;
    border-left: 5px solid #4f46e5;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 14px;
    background: #ffffff;
}
.rule-title { font-size: 14px; font-weight: 700; color: #4f46e5; margin-bottom: 8px; }
.rule-body  { font-size: 16px; font-weight: 400; color: #111111; line-height: 1.6; }
.rule-meta  { font-size: 12px; font-weight: 500; color: #888888; margin-top: 12px; }

/* Prob row */
.prob-row { display: flex; align-items: center; gap: 14px; margin: 10px 0; }
.prob-id   { font-size: 14px; font-weight: 600; color: #888888; width: 22px; }
.prob-name { font-size: 14px; font-weight: 600; color: #111111; width: 180px; }
.prob-bg   { flex: 1; height: 8px; background: #f0f0f0; border-radius: 4px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 4px; background: #15803d; }
.prob-pct  { font-size: 14px; font-weight: 700; color: #111111; width: 42px; text-align: right; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Tabs */
div[data-testid="stTabs"] button {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #555555 !important;
    padding: 10px 20px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #111111 !important;
    border-bottom: 3px solid #111111 !important;
}
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
    'TLT':'#1d4ed8','LQD':'#7c3aed','HYG':'#b45309',
    'VNQ':'#0e7490','GLD':'#ca8a04','SLV':'#6b7280','CASH':'#d1d5db',
}

def regime_pill(name):
    n = (name or '').lower()
    if any(k in n for k in ['crisis','stress','risk off']): cls='pill-red'
    elif any(k in n for k in ['expansion','growth','recovery']): cls='pill-green'
    elif any(k in n for k in ['flat','late','curve']): cls='pill-blue'
    else: cls='pill-gray'
    return f'<span class="pill {cls}">{name}</span>'

def pct(v, d=1):
    if v is None or (isinstance(v,float) and np.isnan(v)): return '—'
    return f"{'+'if v>=0 else ''}{v*100:.{d}f}%"

def flt(v, d=2):
    if v is None or (isinstance(v,float) and np.isnan(v)): return '—'
    return f"{v:.{d}f}"

def vcolor(v, good_positive=True):
    if v is None or (isinstance(v,float) and np.isnan(v)): return ''
    if good_positive: return 'green' if v>=0 else 'red'
    return 'red' if v<0 else 'green'

def chart_equity(history):
    scored = [s for s in history if s.get('scored')]
    if len(scored)<2: return None
    dates = [s['date'][:10] for s in scored]  # Take only YYYY-MM-DD
    pc = (1+np.array([s['portfolio_return'] for s in scored])).cumprod()*100
    bc = (1+np.array([s['benchmark_return'] for s in scored])).cumprod()*100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=pc,name='REALM',
        line=dict(color='#15803d',width=2.5),
        fill='tozeroy',fillcolor='rgba(21,128,61,0.06)'))
    fig.add_trace(go.Scatter(x=dates,y=bc,name='AGG',
        line=dict(color='#aaaaaa',width=1.5,dash='dot')))
    fig.update_layout(paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',
        height=260,margin=dict(l=0,r=0,t=8,b=0),
        font=dict(family='Inter',color='#555555',size=12),
        legend=dict(orientation='h',y=1.12,font=dict(size=13)),
        xaxis=dict(gridcolor='#f0f0f0'),
        yaxis=dict(gridcolor='#f0f0f0',title='Base 100'))
    return fig

def chart_excess(history):
    scored = [s for s in history if s.get('scored')][-30:]
    if not scored: return None
    dates = [s['date'][:10] for s in scored]
    vals  = [s['excess_return']*100 for s in scored]
    colors = ['#15803d' if v>=0 else '#b91c1c' for v in vals]
    fig = go.Figure(go.Bar(x=dates,y=vals,marker_color=colors,marker_line_width=0))
    fig.add_hline(y=0,line_color='#cccccc',line_width=1)
    fig.update_layout(paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',
        height=220,margin=dict(l=0,r=0,t=8,b=0),
        font=dict(family='Inter',color='#555555',size=12),
        xaxis=dict(gridcolor='#f0f0f0',tickangle=-45),
        yaxis=dict(gridcolor='#f0f0f0',ticksuffix='%'),
        showlegend=False)
    return fig

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    signal        = load_signal()
    history       = load_history()
    perf          = load_perf()
    rulebook_data = load_rulebook()
    regime_hist   = load_regime_hist()

    # ── Header ────────────────────────────────────────────────────────────────
    now = datetime.utcnow().strftime('%Y-%m-%d  %H:%M UTC')
    c_title, c_time = st.columns([3,1])
    with c_title:
        st.markdown("""
        <div style="margin-bottom:8px">
            <span style="font-size:32px;font-weight:800;color:#111111;letter-spacing:-0.02em">REALM</span>
            <span style="font-size:15px;font-weight:600;color:#888888;margin-left:16px;letter-spacing:0.06em">P2 ETF ERL ENGINE</span>
        </div>
        <div style="font-size:15px;color:#555555;font-weight:400">
            Regime-Aware Experiential Asset Learning Machine
        </div>
        """, unsafe_allow_html=True)
    with c_time:
        st.markdown(f"""
        <div style="text-align:right;margin-top:10px;font-size:13px;
                    font-weight:500;color:#888888">{now}</div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    if not signal:
        st.error("No signal available yet.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["  Signal  ", "  Regime  ", "  Performance  ", "  Rulebook  "])

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        rname = signal.get('regime_name','Unknown')
        kf    = signal.get('kelly_fraction') or 0
        cp    = signal.get('crisis_prob') or 0
        rs    = signal.get('rolling_sharpe') or 0
        if np.isnan(kf): kf=0
        if np.isnan(cp): cp=0
        if np.isnan(rs): rs=0

        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="card">
                <div class="card-label">Current Regime</div>
                <div style="font-size:24px;font-weight:700;color:#111111">{rname}</div>
                {regime_pill(rname)}
            </div>""", unsafe_allow_html=True)
        with c2:
            kc = 'green' if kf>=0.2 else 'amber' if kf>=0.1 else 'red'
            st.markdown(f"""<div class="card">
                <div class="card-label">Kelly Fraction</div>
                <div class="card-value {kc}">{kf:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            cc = 'red' if cp>0.4 else 'amber' if cp>0.2 else 'green'
            st.markdown(f"""<div class="card">
                <div class="card-label">Crisis Probability</div>
                <div class="card-value {cc}">{cp:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            rc = 'green' if rs>=1 else 'amber' if rs>=0 else 'red'
            st.markdown(f"""<div class="card">
                <div class="card-label">Rolling Sharpe</div>
                <div class="card-value {rc}">{rs:.2f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        cl, cr = st.columns([1.3, 1])

        with cl:
            st.markdown('<div class="sec-head">Portfolio Allocation</div>', unsafe_allow_html=True)
            alloc = signal.get('allocation', {})
            rows  = ''
            for asset, w in sorted(alloc.items(), key=lambda x: -x[1]):
                color = ASSET_COLORS.get(asset, '#15803d')
                rows += f"""<div class="alloc-row">
                    <span class="alloc-ticker">{asset}</span>
                    <div class="alloc-bg">
                        <div class="alloc-fill" style="width:{w*100:.1f}%;background:{color}"></div>
                    </div>
                    <span class="alloc-pct">{w*100:.1f}%</span>
                </div>"""
            st.markdown(rows, unsafe_allow_html=True)
            turn = signal.get('turnover',0) or 0
            top  = signal.get('top_asset','—')
            date = signal.get('date','—')
            st.markdown(f"""<div style="font-size:13px;color:#888888;margin-top:14px;font-weight:500">
                Turnover: {turn:.1%} &nbsp;·&nbsp; Top Asset: {top} &nbsp;·&nbsp; Signal Date: {date}
            </div>""", unsafe_allow_html=True)

        with cr:
            st.markdown('<div class="sec-head">Kelly Sizing</div>', unsafe_allow_html=True)
            rs_sc = signal.get('kelly_regime_scalar',0) or 0
            ag_sc = signal.get('kelly_agreement_scalar',0) or 0
            sh_sc = signal.get('kelly_sharpe_scalar',0) or 0
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

            gA = signal.get('gate_A',0) or 0
            gB = signal.get('gate_B',0) or 0
            gC = signal.get('gate_C',0) or 0
            st.markdown(f"""
            <div class="sec-head" style="margin-top:28px">Ensemble Gates</div>
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
                st.markdown(f"""<div class="rule-card">
                    <div class="rule-body">{r}</div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # REGIME
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        c1, c2 = st.columns([1, 1.8])
        with c1:
            st.markdown('<div class="sec-head">Regime Probabilities</div>', unsafe_allow_html=True)
            hmm = signal.get('hmm_probs', {})
            for k in range(cfg.HMM_N_STATES):
                p  = float(hmm.get(str(k), 0))
                nm = cfg.REGIME_NAMES.get(k, f'Regime {k}')
                st.markdown(f"""<div class="prob-row">
                    <span class="prob-id">{k}</span>
                    <span class="prob-name">{nm}</span>
                    <div class="prob-bg">
                        <div class="prob-fill" style="width:{p*100:.1f}%"></div>
                    </div>
                    <span class="prob-pct">{p:.0%}</span>
                </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="sec-head">Regime Timeline — Last Year</div>',
                        unsafe_allow_html=True)
            if regime_hist is not None and len(regime_hist)>0:
                recent  = regime_hist.tail(252)
                palette = ['#15803d','#7c3aed','#b45309','#0e7490',
                           '#b91c1c','#db2777','#6b7280','#ca8a04']
                fig = go.Figure()
                for k in range(cfg.HMM_N_STATES):
                    mask = recent['regime']==k
                    if mask.sum()==0: continue
                    fig.add_trace(go.Scatter(
                        x=recent[mask]['date'], y=[1]*mask.sum(), mode='markers',
                        marker=dict(symbol='square',size=9,color=palette[k]),
                        name=cfg.REGIME_NAMES.get(k,f'R{k}'),
                    ))
                fig.update_layout(
                    paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                    height=130, margin=dict(l=0,r=0,t=8,b=0),
                    font=dict(family='Inter',color='#555555',size=12),
                    xaxis=dict(gridcolor='#f0f0f0'),
                    yaxis=dict(visible=False),
                    legend=dict(orientation='h',y=1.5,font=dict(size=12)),
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar':False})
            else:
                st.info("Regime history not available yet.")

            st.markdown('<div class="sec-head" style="margin-top:24px">Transition Entropy</div>',
                        unsafe_allow_html=True)
            ent   = signal.get('transition_entropy',0) or 0
            max_e = np.log(cfg.HMM_N_STATES)
            e_pct = ent/max_e if max_e>0 else 0
            ec    = 'red' if e_pct>0.7 else 'amber' if e_pct>0.4 else 'green'
            st.markdown(f"""<div class="card" style="margin-top:0">
                <div class="card-label">Entropy &nbsp;(0 = certain, 1 = maximum uncertainty)</div>
                <div class="card-value {ec}" style="font-size:32px">{ent:.3f}
                    <span style="font-size:18px;color:#888888;font-weight:500">
                        &nbsp;/ {max_e:.2f} &nbsp;({e_pct:.0%})
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        if perf and not perf.get('insufficient_data'):
            c1,c2,c3,c4 = st.columns(4)
            for col, label, val, gp in [
                (c1,"Ann. Excess Return", perf.get('ann_excess_return'), True),
                (c2,"Excess Sharpe",      perf.get('excess_sharpe') if perf.get('n_scored',0)>=5 else None, None),
                (c3,"Win Rate vs AGG",    perf.get('win_rate_vs_bench'), True),
                (c4,"Max Drawdown",       perf.get('max_drawdown') if perf.get('n_scored',0)>=5 else None, False),
            ]:
                with col:
                    if label=="Excess Sharpe":
                        disp=flt(val); c='green' if val and val>=0.5 else 'amber' if val and val>=0 else 'red'
                    elif label=="Max Drawdown":
                        disp=pct(val); c='red' if val and val<-0.05 else 'amber' if val and val<-0.02 else 'green'
                    else:
                        disp=pct(val); c=vcolor(val,gp)
                    st.markdown(f"""<div class="card">
                        <div class="card-label">{label}</div>
                        <div class="card-value {c}">{disp}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            n = perf.get('n_scored', 0) if perf else 0
            st.info(f"Only {n} scored day(s) so far — Sharpe and drawdown metrics require at least 5 days.")

        ec = chart_equity(history)
        if ec:
            st.markdown('<div class="sec-head">Equity Curve (Base 100)</div>',unsafe_allow_html=True)
            st.plotly_chart(ec,use_container_width=True,config={'displayModeBar':False})

        eb = chart_excess(history)
        if eb:
            st.markdown('<div class="sec-head">Daily Excess Return vs AGG — Last 30 Days</div>',unsafe_allow_html=True)
            st.plotly_chart(eb,use_container_width=True,config={'displayModeBar':False})

        if perf and not perf.get('insufficient_data'):
            rp = perf.get('regime_performance',{})
            if rp:
                st.markdown('<div class="sec-head">Performance by Regime</div>',unsafe_allow_html=True)
                rows = [{'Regime':k,'Days':v['n_days'],
                         'Mean Daily Excess':f"{v['mean_excess']*100:+.3f}%",
                         'Win Rate':f"{v['win_rate']:.0%}"}
                        for k,v in rp.items()]
                st.dataframe(pd.DataFrame(rows).sort_values('Days',ascending=False),
                             hide_index=True,use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RULEBOOK
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        if not rulebook_data:
            st.info("No rulebook yet.")
            return

        rules = rulebook_data.get('rules',[])
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="card">
                <div class="card-label">Rules Stored</div>
                <div class="card-value blue">{len(rules)}/{cfg.ERL_MAX_RULES}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            stored=rulebook_data.get('total_stored',0)
            rejected=rulebook_data.get('total_rejected',0)
            total=stored+rejected
            gpr=stored/total if total>0 else 0
            st.markdown(f"""<div class="card">
                <div class="card-label">Gate Pass Rate</div>
                <div class="card-value {'green' if gpr>=0.5 else 'amber'}">{gpr:.0%}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            upd=rulebook_data.get('last_updated','')[:10]
            st.markdown(f"""<div class="card">
                <div class="card-label">Last Updated</div>
                <div class="card-value" style="font-size:24px">{upd}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            gn=sum(1 for r in rules if r.get('source')=='gemini')
            st.markdown(f"""<div class="card">
                <div class="card-label">Gemini Rules</div>
                <div class="card-value">{gn}/{len(rules)}</div>
            </div>""", unsafe_allow_html=True)

        all_regimes = sorted(set(r.get('regime_name','Unknown') for r in rules))
        sel = st.selectbox("Filter by regime", ['All']+all_regimes, label_visibility='collapsed')
        filtered = rules if sel=='All' else [r for r in rules if r.get('regime_name')==sel]
        filtered = sorted(filtered, key=lambda r: -r.get('confidence',0))

        st.markdown(f"""<div style="font-size:14px;color:#888888;margin:16px 0 24px;font-weight:500">
            Showing {len(filtered)} rule(s)
        </div>""", unsafe_allow_html=True)

        for r in filtered:
            conf  = r.get('confidence',0)
            impr  = (r.get('improvement',0) or 0)*10000
            src   = r.get('source','?').upper()
            rtype = r.get('rule_type','?').upper()
            asset = r.get('primary_asset','?')
            rname2= r.get('regime_name','?')
            action= r.get('action', r.get('rationale',''))
            stored= r.get('stored_at','')[:10]
            ep    = r.get('episode_id','?')
            bc    = '#15803d' if conf>=0.7 else '#b45309' if conf>=0.5 else '#888888'

            st.markdown(f"""<div class="rule-card" style="border-left-color:{bc}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                    <span class="rule-title" style="color:{bc}">{rtype} · {asset}</span>
                    <span style="font-size:14px;font-weight:600;color:#555555">{rname2}</span>
                </div>
                <div class="rule-body">{action}</div>
                <div class="rule-meta">
                    Confidence: {conf:.2f} &nbsp;·&nbsp;
                    Improvement: +{impr:.1f} bp &nbsp;·&nbsp;
                    Source: {src} &nbsp;·&nbsp;
                    Episode: {ep} &nbsp;·&nbsp;
                    Stored: {stored}
                </div>
            </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

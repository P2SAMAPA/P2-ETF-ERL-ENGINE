# ═══════════════════════════════════════════════════════════════
# KAGGLE MONDAY NOTEBOOK — copy each block into a separate cell
# Title: p2-etf-erl-train-monday
# GPU:   T4 x2
# ═══════════════════════════════════════════════════════════════


# ── CELL 1 — Install dependencies ──────────────────────────────
import subprocess, sys

subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q',
    'huggingface_hub>=0.20.0',
    'hmmlearn>=0.3.0',
    'google-generativeai>=0.4.0',
], check=True)
print('Dependencies installed')


# ── CELL 2 — Clone repo ─────────────────────────────────────────
import subprocess, sys, os

repo_path = '/kaggle/working/repo'
if not os.path.exists(repo_path):
    subprocess.run([
        'git', 'clone',
        'https://github.com/P2SAMAPA/P2-ETF-ERL-ENGINE.git',
        repo_path
    ], check=True)
else:
    subprocess.run(['git', '-C', repo_path, 'pull'], check=True)

sys.path.insert(0, repo_path)
os.chdir(repo_path)
print(f'Repo ready at {repo_path}')


# ── CELL 3 — Load secrets ───────────────────────────────────────
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ['HF_TOKEN']        = secrets.get_secret('HF_TOKEN')
os.environ['HF_SOURCE_REPO']  = secrets.get_secret('HF_SOURCE_REPO')
os.environ['HF_MODELS_REPO']  = secrets.get_secret('HF_MODELS_REPO')
os.environ['HF_RESULTS_REPO'] = secrets.get_secret('HF_RESULTS_REPO')
os.environ['GEMINI_API_KEY']  = secrets.get_secret('GEMINI_API_KEY')
print('Secrets loaded')


# ── CELL 4 — Validate config ────────────────────────────────────
import config as cfg
cfg.validate()


# ── CELL 5 — HMM Training (~5 min, CPU) ────────────────────────
from hmm_train import main as hmm_main
hmm_main()


# ── CELL 6 — TFT Training (~3 hrs, GPU) ────────────────────────
from tft_train import main as tft_main
tft_main()


# ── CELL 7 — DDPG Training: Policies A, B, C (~4 hrs, GPU) ────
from ddpg_train import main as ddpg_main
ddpg_main()


# ── CELL 8 — ERL Training (~6 hrs, GPU) ────────────────────────
from erl_train import main as erl_main
erl_main()


# ── CELL 9 — Print summaries ────────────────────────────────────
import json, os
import config as cfg

for fname in ['ddpg_summary.json', 'erl_summary.json', 'sft_summary.json']:
    path = os.path.join(cfg.LOCAL_TMP, fname)
    if os.path.exists(path):
        with open(path) as f:
            print(f'\n── {fname} ──')
            print(json.dumps(json.load(f), indent=2))

print('\n✅ Monday training complete')

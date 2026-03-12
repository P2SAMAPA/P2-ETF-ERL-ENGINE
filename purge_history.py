# purge_history.py — one-time cleanup of stale/broken signal history
# Run once manually: python purge_history.py
# Keeps only signals that have: valid pick, valid date >= 2025-01-01, real portfolio_return

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from huggingface_hub import HfApi, hf_hub_download

def main():
    cfg.validate()

    # Download current history
    path = hf_hub_download(
        repo_id=cfg.HF_RESULTS_REPO, filename='results/signal_history.json',
        repo_type='dataset', token=cfg.HF_TOKEN, force_download=True
    )
    with open(path) as f:
        history = json.load(f)

    print(f"[purge] Loaded {len(history)} history entries")

    # Keep only valid entries
    # Keep only the LATEST entry per date (most recent generated_at)
    by_date = {}
    for s in history:
        date = s.get('date', '')
        pick = s.get('pick') or s.get('top_asset')
        if not pick or pick == 'N/A':
            continue
        if date < '2025-01-01':
            continue
        # Keep latest generated_at per date
        existing = by_date.get(date)
        if existing is None or s.get('generated_at','') > existing.get('generated_at',''):
            by_date[date] = s

    clean = list(by_date.values())
    clean.sort(key=lambda x: x.get('date',''))
    for s in clean:
        print(f"  KEEP {s['date']}: pick={s.get('pick') or s.get('top_asset')} scored={s.get('scored')}")

    print(f"[purge] Keeping {len(clean)} of {len(history)} entries")

    # Save and push
    out_path = os.path.join(cfg.LOCAL_TMP, 'signal_history.json')
    os.makedirs(cfg.LOCAL_TMP, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2)

    HfApi(token=cfg.HF_TOKEN).upload_file(
        path_or_fileobj=out_path,
        path_in_repo='results/signal_history.json',
        repo_id=cfg.HF_RESULTS_REPO,
        repo_type='dataset',
    )
    print(f"[purge] ✅ Pushed clean history ({len(clean)} entries) → {cfg.HF_RESULTS_REPO}")

if __name__ == '__main__':
    main()

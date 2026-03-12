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
    clean = []
    for s in history:
        date       = s.get('date','')
        pick       = s.get('pick') or s.get('top_asset')
        port_ret   = s.get('portfolio_return')
        scored_at  = s.get('scored_at', '')
        is_scored  = s.get('scored', False)

        # Must have valid pick
        if not pick or pick == 'N/A':
            print(f"  REMOVE {date}: no valid pick")
            continue
        # Must be in live period
        if date < '2025-01-01':
            print(f"  REMOVE {date}: before live start")
            continue
        # If scored, must have been scored recently (after 2026-03-10) with real return
        if is_scored:
            if port_ret is None:
                print(f"  REMOVE {date}: scored but no return value")
                continue
            if scored_at < '2026-03-10':
                print(f"  REMOVE {date}: scored_at={scored_at[:10]} too old — ghost entry")
                continue
        clean.append(s)

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

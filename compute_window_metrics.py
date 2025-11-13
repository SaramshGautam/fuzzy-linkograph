from __future__ import annotations
import json, numpy as np, datetime as dt, pandas as pd
from math import log2
from pathlib import Path

# ==== CONFIG ====
WINDOW_BASE_SEC = 30
MIN_LINK_STRENGTH = 0.35
OUT_CSV = "teamx_window_metrics.csv"

def entropy(p_on):
    """Binary entropy of link existence."""
    p_off = 1 - p_on
    if p_on == 0 or p_on == 1: return 0
    return -(p_on*log2(p_on) + p_off*log2(p_off))

def total_link_weight(vals):
    """Sum of link weights after threshold/scale."""
    vals = [v for v in vals if v >= MIN_LINK_STRENGTH]
    if not vals: return 0
    scaled = [(v - MIN_LINK_STRENGTH) / (1 - MIN_LINK_STRENGTH) for v in vals]
    return sum(scaled)

def compute_metrics(episode):
    moves, links = episode["moves"], episode["links"]

    # Convert timestamps to datetimes
    for m in moves:
        m["_t"] = dt.datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
    t0, tN = moves[0]["_t"], moves[-1]["_t"]
    dur = (tN - t0).total_seconds()
    n_windows = int(np.ceil(dur / WINDOW_BASE_SEC))
    windows = []
    for w in range(n_windows):
        t_start = t0 + dt.timedelta(seconds=w * WINDOW_BASE_SEC)
        t_end = t_start + dt.timedelta(seconds=WINDOW_BASE_SEC)
        win_moves = [m for m in moves if t_start <= m["_t"] < t_end]
        if not win_moves: continue

        # compute link-based measures
        idxs = [moves.index(m) for m in win_moves]
        fore = []
        back = []
        for i in idxs:
            back.extend(links.get(str(i), {}).values())
            for j in range(i+1, len(moves)):
                if str(j) in links and str(i) in links[str(j)]:
                    fore.append(links[str(j)][str(i)])
        fore_weight = total_link_weight(fore)
        back_weight = total_link_weight(back)
        total_links = fore_weight + back_weight
        link_density = total_links / max(1, len(win_moves))
        fore_ratio = fore_weight / (total_links + 1e-6)
        back_ratio = back_weight / (total_links + 1e-6)

        # entropy estimate
        if total_links > 0:
            e_val = entropy(fore_ratio)
        else:
            e_val = 0

        # micro-behavior summaries
        dwell = np.mean([m["micro"]["dwell_ms"] for m in win_moves])
        hover = np.mean([m["micro"]["hover_ms"] for m in win_moves])
        apm = np.mean([m["tempo"]["apm_rolling"] for m in win_moves])
        gap = np.mean([m["tempo"]["gap_prev_ms"] for m in win_moves])

        windows.append({
            "window_id": w,
            "t_start": t_start.isoformat(),
            "t_end": t_end.isoformat(),
            "num_moves": len(win_moves),
            "fore_weight": fore_weight,
            "back_weight": back_weight,
            "link_density": link_density,
            "fore_ratio": fore_ratio,
            "back_ratio": back_ratio,
            "entropy": e_val,
            "dwell_ms": dwell,
            "hover_ms": hover,
            "apm_rolling": apm,
            "gap_prev_ms": gap,
        })
    return windows

def main(fpath="./data/TeamD_15min_augmented_linked.json"):
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_rows = []
    for ep_id, episode in data.items():
        print("Episode:", ep_id)
        ws = compute_metrics(episode)
        for w in ws:
            w["episode"] = ep_id
        all_rows.extend(ws)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved metrics to {OUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-end Linkograph Pipeline

Usage:
  python linkograph_pipeline.py \
    --input teamx_session_10min.json \
    --outdir outputs \
    --window-sec 30 \
    --min-link 0.35 \
    --apm-low 12 --apm-high 20 --gap-high 1200 \
    --fore-thresh 0.60 --back-thresh 0.60 \
    --smooth-radius 1

Outputs (in --outdir):
  - window metrics:          teamx_window_metrics.csv
  - labeled window metrics:  teamx_window_metrics_labeled.csv
  - phase confidence:        teamx_phase_confidence.csv
  - chart:                   teamx_phase_confidence_lines.png
  - chart:                   teamx_phase_confidence_timeline.png
"""

from __future__ import annotations
import argparse, json, os, sys, re, time, datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: only needed for the embedding step
try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    SentenceTransformer = None
    util = None


# -----------------------------
# Helpers
# -----------------------------

def iso_to_dt(s: str) -> dt.datetime:
    """Parse ISO 8601 with or without 'Z' into a timezone-aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return dt.datetime.fromisoformat(s)


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_load_any(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump_pretty(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def bincount_majority(arr: np.ndarray, num_classes: int, tie_break: int) -> int:
    counts = np.bincount(arr, minlength=num_classes)
    maj = int(np.argmax(counts))
    return maj if counts[maj] != 0 else tie_break


# -----------------------------
# 1) Load moves / episodes
# -----------------------------

def load_episodes(input_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Supports two common formats:
    A) {"TeamX": [ {move}, {move}, ... ]}
    B) {"TeamX": {"moves": [...], "links": {...} } }
    Returns: { episode_id: {"moves":[...], "links": None or {...}} }
    """
    raw = json_load_any(input_path)
    episodes = {}
    for ep_id, payload in raw.items():
        if isinstance(payload, list):
            episodes[ep_id] = {"moves": payload, "links": None}
        elif isinstance(payload, dict) and "moves" in payload:
            episodes[ep_id] = {"moves": payload["moves"], "links": payload.get("links")}
        else:
            raise ValueError(f"Unrecognized episode format under key '{ep_id}'")
    return episodes


# -----------------------------
# 2) Compute links (embeddings + cosine sim)
# -----------------------------

def compute_links(moves: List[Dict[str, Any]], model_name: str, batch_size: int = 64) -> Dict[str, Dict[str, float]]:
    if SentenceTransformer is None or util is None:
        raise RuntimeError("sentence_transformers is required for link computation. Please install it locally.")
    model = SentenceTransformer(model_name)
    texts = [m.get("text", "") for m in moves]
    # Normalize embeddings for cosine
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
    n = len(embs)
    links: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        links[str(i)] = {}
        for j in range(i):
            sim = float(np.dot(embs[i], embs[j]))  # cosine since normalized
            links[str(i)][str(j)] = sim
    return links


# -----------------------------
# 3) Window metrics
# -----------------------------

def total_link_weight(vals: List[float], min_link: float) -> float:
    """Sum of link weights after threshold + linear rescale into [0,1]."""
    good = [v for v in vals if v >= min_link]
    if not good:
        return 0.0
    scaled = [(v - min_link) / (1.0 - min_link) for v in good]
    return float(sum(scaled))


def compute_window_metrics(moves: List[Dict[str, Any]], links: Dict[str, Dict[str, float]], window_sec: int, min_link: float) -> pd.DataFrame:
    # timestamps to datetime
    for m in moves:
        m["_t"] = iso_to_dt(m["timestamp"])

    t0, tN = moves[0]["_t"], moves[-1]["_t"]
    dur = (tN - t0).total_seconds()
    n_windows = int(np.ceil(dur / window_sec))

    rows = []
    for w in range(n_windows):
        t_start = t0 + dt.timedelta(seconds=w * window_sec)
        t_end = t_start + dt.timedelta(seconds=window_sec)
        win_moves = [m for m in moves if t_start <= m["_t"] < t_end]
        if not win_moves:
            continue

        # indexes of moves in full list
        idxs = [moves.index(m) for m in win_moves]

        fore_vals: List[float] = []
        back_vals: List[float] = []
        for i in idxs:
            # backlinks (j -> i, j<i)
            back_vals.extend(list(links.get(str(i), {}).values()))
            # forelinks (i -> j as stored in links[j][i])
            for j in range(i + 1, len(moves)):
                v = links.get(str(j), {}).get(str(i), 0.0)
                fore_vals.append(v)

        fore_weight = total_link_weight(fore_vals, min_link)
        back_weight = total_link_weight(back_vals, min_link)
        total_w = fore_weight + back_weight

        link_density = total_w / max(1, len(win_moves))
        fore_ratio = fore_weight / (total_w + 1e-6)
        back_ratio = back_weight / (total_w + 1e-6)

        # Micro-behavior rollups (use 0 if missing)
        def avg(field_chain: List[str], default: float = 0.0) -> float:
            vals = []
            for m in win_moves:
                cur = m
                ok = True
                for f in field_chain:
                    if isinstance(cur, dict) and f in cur:
                        cur = cur[f]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, (int, float)):
                    vals.append(float(cur))
            return float(np.mean(vals)) if vals else default

        dwell = avg(["micro", "dwell_ms"])
        hover = avg(["micro", "hover_ms"])
        apm = avg(["tempo", "apm_rolling"])
        gap = avg(["tempo", "gap_prev_ms"])

        rows.append({
            "window_id": w,
            "t_start": t_start.isoformat(),
            "t_end": t_end.isoformat(),
            "num_moves": len(win_moves),
            "fore_weight": fore_weight,
            "back_weight": back_weight,
            "link_density": link_density,
            "fore_ratio": fore_ratio,
            "back_ratio": back_ratio,
            "dwell_ms": dwell,
            "hover_ms": hover,
            "apm_rolling": apm,
            "gap_prev_ms": gap,
        })

    return pd.DataFrame(rows)


# -----------------------------
# 4) Phase rules + smoothing
# -----------------------------

def rule_label(row, fore_thresh: float, back_thresh: float, apm_high: float, apm_low: float, gap_high_ms: float) -> str:
    fore = float(row.get("fore_ratio", 0.0))
    back = float(row.get("back_ratio", 0.0))
    apm  = float(row.get("apm_rolling", 0.0))
    gap  = float(row.get("gap_prev_ms", 0.0))

    if apm < apm_low or gap >= gap_high_ms:
        return "incubation"
    if fore >= fore_thresh and apm >= apm_high and fore > back:
        return "divergent"
    if back >= back_thresh and apm >= (apm_high - 5) and back > fore:
        return "convergent"
    return "conflict"


def smooth_series(phases: List[str], radius: int) -> List[str]:
    mapping = {"incubation": 0, "divergent": 1, "convergent": 2, "conflict": 3}
    inv = {v: k for k, v in mapping.items()}
    ints = np.array([mapping.get(p, 3) for p in phases], dtype=int)
    out = []
    for i in range(len(ints)):
        lo = max(0, i - radius)
        hi = min(len(ints), i + radius + 1)
        w = ints[lo:hi]
        maj = np.bincount(w, minlength=len(mapping)).argmax()
        out.append(inv[int(maj)])
    return out


# -----------------------------
# 5) Confidence scores
# -----------------------------

def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def smoothstep(x: float) -> float:
    x = clip01(x)
    return x * x * (3 - 2 * x)


def z01(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return clip01((x - lo) / (hi - lo))


def compute_confidence(df: pd.DataFrame, apm_low: float, apm_high: float, gap_high_ms: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        fore = float(r.get("fore_ratio", 0.0))
        back = float(r.get("back_ratio", 0.0))
        apm  = float(r.get("apm_rolling", 0.0))
        gap  = float(r.get("gap_prev_ms", 0.0))
        dens = float(r.get("link_density", 0.0))

        activity = z01(apm, apm_low, apm_high)
        rest = clip01(gap / (gap_high_ms * 1.5))

        fore_margin = clip01((fore - back + 1.0) / 2.0)
        div_score = smoothstep(fore) * smoothstep(activity) * smoothstep(fore_margin) * (0.7 + 0.3 * clip01(dens))

        back_margin = clip01((back - fore + 1.0) / 2.0)
        conv_score = smoothstep(back) * smoothstep(activity) * smoothstep(back_margin) * (0.7 + 0.3 * clip01(dens))

        inc_score = smoothstep(1.0 - activity) * smoothstep(rest) * smoothstep(1.0 - max(fore, back))

        conflict_symmetry = 1.0 - abs(fore - back)
        conflict_act = smoothstep(activity)
        conf_score = smoothstep(conflict_symmetry) * conflict_act * (0.6 + 0.4 * (1.0 - inc_score))

        arr = np.array([div_score, conv_score, inc_score, conf_score], dtype=float)
        if arr.sum() == 0:
            arr = np.ones_like(arr) / len(arr)
        else:
            arr = arr / arr.sum()

        labels = ["divergent", "convergent", "incubation", "conflict"]
        pred_idx = int(np.argmax(arr))
        pred = labels[pred_idx]
        conf = float(arr[pred_idx])

        rows.append({
            "window_id": int(r["window_id"]),
            "p_divergent": float(arr[0]),
            "p_convergent": float(arr[1]),
            "p_incubation": float(arr[2]),
            "p_conflict": float(arr[3]),
            "predicted_phase": pred,
            "confidence": conf,
        })
    return pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)


# -----------------------------
# 6) Plotting
# -----------------------------

def plot_confidence_lines(conf_df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 4))
    x = conf_df["window_id"]
    plt.plot(x, conf_df["p_divergent"], label="p_divergent")
    plt.plot(x, conf_df["p_convergent"], label="p_convergent")
    plt.plot(x, conf_df["p_incubation"], label="p_incubation")
    plt.plot(x, conf_df["p_conflict"], label="p_conflict")
    plt.ylim(0, 1.05)
    plt.xlabel("window_id")
    plt.ylabel("probability")
    plt.title("Per-Phase Confidence by Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_top_phase_timeline(conf_df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 2.2))
    x = conf_df["window_id"]
    y = conf_df["confidence"]
    plt.bar(x, y)
    for i, (xi, ph) in enumerate(zip(x, conf_df["predicted_phase"])):
        plt.text(xi, y.iloc[i] + 0.02, ph[:1].upper(), ha="center", va="bottom", fontsize=8)
    plt.ylim(0, 1.15)
    plt.xlabel("window_id")
    plt.ylabel("confidence")
    plt.title("Top-Phase Confidence Timeline")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="End-to-end linkograph pipeline")
    p.add_argument("--input", required=True, help="Path to session JSON (e.g., teamx_session_10min.json)")
    p.add_argument("--outdir", default="outputs", help="Directory to write outputs")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model id")
    p.add_argument("--window-sec", type=int, default=30, help="Window size in seconds")
    p.add_argument("--min-link", type=float, default=0.35, help="Minimum link strength to count")
    p.add_argument("--fore-thresh", type=float, default=0.60, help="Forelink ratio threshold for divergence")
    p.add_argument("--back-thresh", type=float, default=0.60, help="Backlink ratio threshold for convergence")
    p.add_argument("--apm-high", type=float, default=20.0, help="APM threshold for high activity")
    p.add_argument("--apm-low", type=float, default=12.0, help="APM threshold for low activity / incubation")
    p.add_argument("--gap-high", type=float, default=1200.0, help="Gap (ms) threshold for incubation")
    p.add_argument("--smooth-radius", type=int, default=1, help="Smoothing radius (windows)")
    args = p.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # 1) Load episodes
    episodes = load_episodes(input_path)

    # For each episode, compute links (if missing), window metrics, labels, confidence
    all_metrics: List[pd.DataFrame] = []
    for ep_id, ep in episodes.items():
        moves = ep["moves"]
        links = ep["links"]
        if not links:
            print(f"[{ep_id}] Computing links with model={args.model} (n={len(moves)}) ...")
            links = compute_links(moves, args.model)
        else:
            print(f"[{ep_id}] Using pre-computed links")

        # 3) Window metrics
        print(f"[{ep_id}] Computing window metrics (window={args.window_sec}s, min_link={args.min_link}) ...")
        df = compute_window_metrics(moves, links, args.window_sec, args.min_link)
        df["episode"] = ep_id
        all_metrics.append(df)

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()

    # Save metrics CSV
    metrics_csv = outdir / "teamx_window_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"✅ Saved window metrics: {metrics_csv}")

    # 4) Phase rules + smoothing
    labeled_df = metrics_df.copy()
    labeled_df["phase_raw"] = labeled_df.apply(
        lambda row: rule_label(
            row,
            fore_thresh=args.fore_thresh,
            back_thresh=args.back_thresh,
            apm_high=args.apm_high,
            apm_low=args.apm_low,
            gap_high_ms=args.gap_high,
        ),
        axis=1,
    )
    # smooth separately per episode to avoid cross-episode bleed
    smoothed = []
    for ep_id, g in labeled_df.groupby("episode", sort=False):
        smoothed.extend(smooth_series(g["phase_raw"].tolist(), radius=args.smooth_radius))
    labeled_df["phase_smooth"] = smoothed

    labeled_csv = outdir / "teamx_window_metrics_labeled.csv"
    labeled_df.to_csv(labeled_csv, index=False)
    print(f"✅ Saved labeled metrics: {labeled_csv}")

    # 5) Confidence
    conf_df = compute_confidence(labeled_df, apm_low=args.apm_low, apm_high=args.apm_high, gap_high_ms=args.gap_high)
    conf_csv = outdir / "teamx_phase_confidence.csv"
    conf_df.to_csv(conf_csv, index=False)
    print(f"✅ Saved phase confidence: {conf_csv}")

    # 6) Plots
    lines_png = outdir / "teamx_phase_confidence_lines.png"
    bar_png   = outdir / "teamx_phase_confidence_timeline.png"
    plot_confidence_lines(conf_df, lines_png)
    plot_top_phase_timeline(conf_df, bar_png)
    print(f"🖼  Saved charts:\n  - {lines_png}\n  - {bar_png}")

    print("Done.")


if __name__ == "__main__":
    main()

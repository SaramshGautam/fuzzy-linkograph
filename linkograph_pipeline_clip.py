#!/usr/bin/env python3
"""
End-to-end Linkograph Pipeline (Text or CLIP Multimodal)

Usage (CLIP multimodal):
  python linkograph_pipeline_clip.py --input teamx_single_episode_separated --outdir outputs -embedder clip -clip-model openai/clip-vit-base-patch32 --fuse-text-weight 0.6 --window-sec 30 --min-link 0.45 --apm-low 12 --apm-high 20 --gap-high 1200 --fore-thresh 0.60 --back-thresh 0.60 --smooth-radius 1
  python linkograph_pipeline_clip.py --input teamx_v2_linked.json --outdir outputs_v2_clip --embedder clip --clip-model openai/clip-vit-base-patch32 --fuse-text-weight 0.6 --window-sec 30 --min-link 0.5


Usage (SentenceTransformers text-only):
  python linkograph_pipeline_multimodal.py \
    --input teamx_session_10min.json \
    --outdir outputs \
    --embedder st \
    --st-model all-MiniLM-L6-v2 \
    --window-sec 30 \
    --min-link 0.35

Outputs in --outdir:
  - teamx_window_metrics.csv
  - teamx_window_metrics_labeled.csv
  - teamx_phase_confidence.csv
  - teamx_phase_confidence_lines.png
  - teamx_phase_confidence_timeline.png
And next to input:
  - <input_basename>_linked.json (precomputed links snapshot)
"""

from __future__ import annotations
import argparse, json, os, re, sys, time, io, math, datetime as dt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Optional imports (lazy used) ----
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# CLIP (only used if --embedder clip)
try:
    import torch
    from PIL import Image
    import requests
    from transformers import CLIPProcessor, CLIPModel
except Exception:
    torch = None
    Image = None
    requests = None
    CLIPProcessor = None
    CLIPModel = None


# =========================
# General Helpers
# =========================
def iso_to_dt(s: str) -> dt.datetime:
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

def text_command_clean(s: str) -> str:
    """Remove UI/action verbs and tidy whitespace."""
    if not isinstance(s, str):
        return ""
    verbs = [
        "add_note","edit","move","cluster","highlight","link","group","draw",
        "create","update","delete","ungroup","select","drag","drop","resize",
        "note","image","item","shape","actor"
    ]
    s = re.sub(r"\b(" + "|".join(verbs) + r")\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_episodes(input_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Accepts:
      A) {"TeamX": [ {move}, ... ]}
      B) {"TeamX": {"moves":[...], "links": {...}}}
    Returns: { ep_id: {"moves":[...], "links": dict|None} }
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


# =========================
# Embedders
# =========================
def compute_links_st(moves: List[Dict[str, Any]], model_name: str, batch_size: int = 64) -> Dict[str, Dict[str, float]]:
    if SentenceTransformer is None:
        raise RuntimeError("sentence_transformers is required for --embedder st")
    model = SentenceTransformer(model_name)
    texts = [text_command_clean(m.get("text", "")) for m in moves]
    embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                        batch_size=batch_size, show_progress_bar=False)
    n = len(embs)
    links: Dict[str, Dict[str, float]] = {}
    for i in range(n):
        links[str(i)] = {}
        for j in range(i):
            links[str(i)][str(j)] = float(np.dot(embs[i], embs[j]))
    return links

# ---- CLIP helpers ----
def _normalize_rows_t(x):
    eps = 1e-8
    norms = torch.clamp(x.norm(dim=1, keepdim=True), min=eps)
    return x / norms

def _download_image(url: str, timeout: int = 15) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

def _first_nonempty(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

TEXT_KEYS = ("text", "label", "content")   # where text might live
URL_KEYS  = ("url", "imageUrl", "src")     # where image URL might live
TYPE_KEYS = ("shapeType", "type", "kind")  # "image" / "text" sometimes here

class CLIPEmbedder:
    def __init__(self, model_id: str, device: Optional[str] = None):
        if torch is None or CLIPModel is None:
            raise RuntimeError("transformers[torch], PIL, and requests are required for --embedder clip")
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.inference_mode()
    def encode_text(self, texts: List[str]):
        if not texts:
            return torch.empty((0, self.model.config.projection_dim), device=self.device)
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        feats = self.model.get_text_features(**inputs)
        return _normalize_rows_t(feats)

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image]):
        if not images:
            return torch.empty((0, self.model.config.projection_dim), device=self.device)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return _normalize_rows_t(feats)

def embed_moves_clip(moves: List[Dict[str, Any]], clipper: CLIPEmbedder, fuse_text_weight: float = 0.6) -> torch.Tensor:
    N = len(moves)
    D = clipper.model.config.projection_dim
    out = torch.zeros((N, D), device=clipper.device)

    text_idx, text_payloads = [], []
    img_idx, img_payloads = [], []

    for i, m in enumerate(moves):
        t = _first_nonempty(m, TEXT_KEYS)
        u = _first_nonempty(m, URL_KEYS)

        if t:
            t = text_command_clean(t)
            if t:
                text_idx.append(i)
                text_payloads.append(t)

        if u:
            img = _download_image(u)
            if img is not None:
                img_idx.append(i)
                img_payloads.append(img)

    t_emb = clipper.encode_text(text_payloads)  if text_payloads else torch.empty((0, D), device=clipper.device)
    i_emb = clipper.encode_images(img_payloads) if img_payloads  else torch.empty((0, D), device=clipper.device)

    tpos = {idx: k for k, idx in enumerate(text_idx)}
    ipos = {idx: k for k, idx in enumerate(img_idx)}

    for i in range(N):
        has_t = i in tpos
        has_i = i in ipos
        if has_t and has_i:
            fused = fuse_text_weight * t_emb[tpos[i]] + (1.0 - fuse_text_weight) * i_emb[ipos[i]]
            fused = fused / (fused.norm() + 1e-8)
            out[i] = fused
        elif has_t:
            out[i] = t_emb[tpos[i]]
        elif has_i:
            out[i] = i_emb[ipos[i]]
        # else remains zero

    return out

def compute_links_clip(moves: List[Dict[str, Any]], clip_model: str, fuse_text_weight: float) -> Dict[str, Dict[str, float]]:
    clipper = CLIPEmbedder(clip_model)
    embs = embed_moves_clip(moves, clipper, fuse_text_weight=fuse_text_weight)
    n = embs.shape[0]
    links: Dict[str, Dict[str, float]] = {str(i): {} for i in range(n)}
    if n == 0:
        return links
    for i in range(n):
        vi = embs[i]
        if torch.allclose(vi, torch.zeros_like(vi)):
            continue
        sims = (embs[:i] @ vi).detach().cpu().numpy()  # cosine for normalized
        for j in range(i):
            links[str(i)][str(j)] = float(sims[j])
    return links


# =========================
# Link Post-Processing
# =========================
def prune_links(links: Dict[str, Dict[str, float]], min_link: float) -> Dict[str, Dict[str, float]]:
    pruned: Dict[str, Dict[str, float]] = {}
    for i, row in links.items():
        keep = {j: float(v) for j, v in row.items() if v >= min_link}
        pruned[i] = keep
    return pruned


# =========================
# Window Metrics
# =========================
def total_link_weight(vals: List[float], min_link: float) -> float:
    good = [v for v in vals if v >= min_link]
    if not good:
        return 0.0
    # linear rescale into [0,1] after threshold
    scaled = [(v - min_link) / (1.0 - min_link) for v in good]
    return float(sum(scaled))

def compute_window_metrics(moves: List[Dict[str, Any]], links: Dict[str, Dict[str, float]],
                           window_sec: int, min_link: float) -> pd.DataFrame:
    for m in moves:
        m["_t"] = iso_to_dt(m["timestamp"])
    t0, tN = moves[0]["_t"], moves[-1]["_t"]
    dur = (tN - t0).total_seconds()
    n_windows = max(1, int(np.ceil(dur / window_sec)))

    rows = []
    for w in range(n_windows):
        t_start = t0 + dt.timedelta(seconds=w * window_sec)
        t_end = t_start + dt.timedelta(seconds=window_sec)
        win_moves = [m for m in moves if t_start <= m["_t"] < t_end]
        if not win_moves:
            continue

        idxs = [moves.index(m) for m in win_moves]
        fore_vals, back_vals = [], []

        for i in idxs:
            # backlinks (j -> i), stored as links[i][j] for j<i
            back_vals.extend(list(links.get(str(i), {}).values()))
            # forelinks (i -> j), found in links[j][i] for j>i
            for j in range(i + 1, len(moves)):
                fore_vals.append(links.get(str(j), {}).get(str(i), 0.0))

        fore_w = total_link_weight(fore_vals, min_link)
        back_w = total_link_weight(back_vals, min_link)
        total_w = fore_w + back_w

        link_density = total_w / max(1, len(win_moves))
        fore_ratio = fore_w / (total_w + 1e-6)
        back_ratio = back_w / (total_w + 1e-6)

        def avg(chain: List[str], default: float = 0.0) -> float:
            vals = []
            for m in win_moves:
                cur, ok = m, True
                for f in chain:
                    if isinstance(cur, dict) and f in cur:
                        cur = cur[f]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, (int, float)):
                    vals.append(float(cur))
            return float(np.mean(vals)) if vals else default

        rows.append({
            "window_id": w,
            "t_start": t_start.isoformat(),
            "t_end": t_end.isoformat(),
            "num_moves": len(win_moves),
            "fore_weight": fore_w,
            "back_weight": back_w,
            "link_density": link_density,
            "fore_ratio": fore_ratio,
            "back_ratio": back_ratio,
            "dwell_ms": avg(["micro","dwell_ms"]),
            "hover_ms": avg(["micro","hover_ms"]),
            "apm_rolling": avg(["tempo","apm_rolling"]),
            "gap_prev_ms": avg(["tempo","gap_prev_ms"]),
        })

    return pd.DataFrame(rows)


# =========================
# Phase Labeling + Smoothing
# =========================
# def rule_label(row, fore_thresh: float, back_thresh: float, apm_high: float, apm_low: float, gap_high_ms: float) -> str:
#     fore = float(row.get("fore_ratio", 0.0))
#     back = float(row.get("back_ratio", 0.0))
#     apm  = float(row.get("apm_rolling", 0.0))
#     gap  = float(row.get("gap_prev_ms", 0.0))
#     if apm < apm_low or gap >= gap_high_ms:
#         return "incubation"
#     if fore >= fore_thresh and apm >= apm_high and fore > back:
#         return "divergent"
#     if back >= back_thresh and apm >= (apm_high - 5) and back > fore:
#         return "convergent"
#     return "conflict"

def rule_label(row, fore_thresh: float, back_thresh: float, apm_high: float, apm_low: float, gap_high_ms: float) -> str:
    fore = float(row.get("fore_ratio", 0.0))
    back = float(row.get("back_ratio", 0.0))
    # two-class decision: whichever ratio is larger
    return "divergent" if fore >= back else "convergent"


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


# =========================
# Confidence Scores
# =========================
def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def smoothstep(x: float) -> float:
    x = clip01(x)
    return x * x * (3 - 2 * x)

def z01(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return clip01((x - lo) / (hi - lo))

# def compute_confidence(df: pd.DataFrame, apm_low: float, apm_high: float, gap_high_ms: float) -> pd.DataFrame:
#     rows = []
#     for _, r in df.iterrows():
#         fore = float(r.get("fore_ratio", 0.0))
#         back = float(r.get("back_ratio", 0.0))
#         apm  = float(r.get("apm_rolling", 0.0))
#         gap  = float(r.get("gap_prev_ms", 0.0))
#         dens = float(r.get("link_density", 0.0))

#         activity = z01(apm, apm_low, apm_high)
#         rest = clip01(gap / (gap_high_ms * 1.5))

#         fore_margin = clip01((fore - back + 1.0) / 2.0)
#         div_score = smoothstep(fore) * smoothstep(activity) * smoothstep(fore_margin) * (0.7 + 0.3 * clip01(dens))

#         back_margin = clip01((back - fore + 1.0) / 2.0)
#         conv_score = smoothstep(back) * smoothstep(activity) * smoothstep(back_margin) * (0.7 + 0.3 * clip01(dens))

#         inc_score = smoothstep(1.0 - activity) * smoothstep(rest) * smoothstep(1.0 - max(fore, back))

#         conflict_symmetry = 1.0 - abs(fore - back)
#         conflict_act = smoothstep(activity)
#         conf_score = smoothstep(conflict_symmetry) * conflict_act * (0.6 + 0.4 * (1.0 - inc_score))

#         arr = np.array([div_score, conv_score, inc_score, conf_score], dtype=float)
#         arr = arr / (arr.sum() if arr.sum() > 0 else 1.0)

#         labels = ["divergent", "convergent", "incubation", "conflict"]
#         pred_idx = int(np.argmax(arr))
#         rows.append({
#             "window_id": int(r["window_id"]),
#             "p_divergent": float(arr[0]),
#             "p_convergent": float(arr[1]),
#             "p_incubation": float(arr[2]),
#             "p_conflict": float(arr[3]),
#             "predicted_phase": labels[pred_idx],
#             "confidence": float(arr[pred_idx]),
#         })
#     return pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)

def compute_confidence(df: pd.DataFrame, apm_low: float, apm_high: float, gap_high_ms: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        fore = float(r.get("fore_ratio", 0.0))
        back = float(r.get("back_ratio", 0.0))
        apm  = float(r.get("apm_rolling", 0.0))
        gap  = float(r.get("gap_prev_ms", 0.0))
        dens = float(r.get("link_density", 0.0))

        # Optional activity weighting (keeps high-activity windows more confident)
        def clip01(x): return float(min(1.0, max(0.0, x)))
        def smoothstep(x): x=clip01(x); return x*x*(3-2*x)
        def z01(x,lo,hi): return 0.0 if hi==lo else clip01((x-lo)/(hi-lo))

        activity = z01(apm, apm_low, apm_high)
        margin   = clip01((fore - back + 1.0) / 2.0)  # >0.5 → fore>back

        # Scores: ratios × activity × density bonus
        div_score = smoothstep(fore) * smoothstep(activity) * (0.7 + 0.3 * clip01(dens))
        con_score = smoothstep(back) * smoothstep(activity) * (0.7 + 0.3 * clip01(dens))

        s = div_score + con_score
        if s == 0: 
            p_div, p_con = 0.5, 0.5
        else:
            p_div, p_con = div_score/s, con_score/s

        pred = "divergent" if p_div >= p_con else "convergent"
        conf = max(p_div, p_con)

        rows.append({
            "window_id": int(r["window_id"]),
            "p_divergent": float(p_div),
            "p_convergent": float(p_con),
            "predicted_phase": pred,
            "confidence": conf,
        })
    return pd.DataFrame(rows).sort_values("window_id").reset_index(drop=True)


# =========================
# Plotting
# =========================
# def plot_confidence_lines(conf_df: pd.DataFrame, out_png: Path) -> None:
#     plt.figure(figsize=(10, 4))
#     x = conf_df["window_id"]
#     plt.plot(x, conf_df["p_divergent"], label="p_divergent")
#     plt.plot(x, conf_df["p_convergent"], label="p_convergent")
#     plt.plot(x, conf_df["p_incubation"], label="p_incubation")
#     plt.plot(x, conf_df["p_conflict"], label="p_conflict")
#     plt.ylim(0, 1.05)
#     plt.xlabel("window_id")
#     plt.ylabel("probability")
#     plt.title("Per-Phase Confidence by Window")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=150)
#     plt.close()


def plot_confidence_lines(conf_df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(10, 4))
    x = conf_df["window_id"]
    plt.plot(x, conf_df["p_divergent"], label="p_divergent")
    plt.plot(x, conf_df["p_convergent"], label="p_convergent")
    plt.ylim(0, 1.05)
    plt.xlabel("window_id")
    plt.ylabel("probability")
    plt.title("D vs C Confidence by Window")
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


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser(description="End-to-end linkograph pipeline (text or CLIP multimodal)")
    p.add_argument("--input", required=True, help="Path to session JSON")
    p.add_argument("--outdir", default="outputs", help="Directory to write outputs")
    p.add_argument("--embedder", choices=["st","clip"], default="clip", help="Embedding backend")
    p.add_argument("--st-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model id (for --embedder st)")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32", help="CLIP model id (for --embedder clip)")
    p.add_argument("--fuse-text-weight", type=float, default=0.6, help="Weight for text when fusing text+image in CLIP (0..1)")
    p.add_argument("--window-sec", type=int, default=30, help="Window size in seconds")
    p.add_argument("--min-link", type=float, default=0.35, help="Minimum link strength to count & to store")
    p.add_argument("--fore-thresh", type=float, default=0.60)
    p.add_argument("--back-thresh", type=float, default=0.60)
    p.add_argument("--apm-high", type=float, default=20.0)
    p.add_argument("--apm-low", type=float, default=12.0)
    p.add_argument("--gap-high", type=float, default=1200.0)
    p.add_argument("--smooth-radius", type=int, default=1)
    args = p.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    episodes = load_episodes(input_path)

    # Compute links (if missing) using requested embedder
    linked_bundle: Dict[str, Any] = {}
    for ep_id, ep in episodes.items():
        moves = ep["moves"]
        links = ep["links"]

        if links is None:
            print(f"[{ep_id}] Computing links via {args.embedder} … (n={len(moves)})")
            if args.embedder == "st":
                links = compute_links_st(moves, args.st_model)
            else:
                links = compute_links_clip(moves, args.clip_model, args.fuse_text_weight)
        else:
            print(f"[{ep_id}] Using pre-computed links")

        # Persist a pruned snapshot for reproducibility
        pruned = prune_links(links, args.min_link)
        linked_bundle[ep_id] = {"moves": moves, "links": pruned}

        # ---- Window metrics
        print(f"[{ep_id}] Computing window metrics (win={args.window_sec}s, min_link={args.min_link})")
        df = compute_window_metrics(moves, pruned, args.window_sec, args.min_link)
        df["episode"] = ep_id

        # label + smooth per-episode
        df["phase_raw"] = df.apply(lambda r: rule_label(
            r,
            fore_thresh=args.fore_thresh,
            back_thresh=args.back_thresh,
            apm_high=args.apm_high,
            apm_low=args.apm_low,
            gap_high_ms=args.gap_high
        ), axis=1)
        df["phase_smooth"] = smooth_series(df["phase_raw"].tolist(), radius=args.smooth_radius)

        # stash per-episode metrics
        if ep_id == list(episodes.keys())[0]:
            all_metrics = df.copy()
        else:
            all_metrics = pd.concat([all_metrics, df], ignore_index=True)

    # Save metrics CSVs
    metrics_csv = outdir / "teamx_window_metrics.csv"
    all_metrics.to_csv(metrics_csv, index=False)
    print(f"✅ Saved window metrics: {metrics_csv}")

    labeled_csv = outdir / "teamx_window_metrics_labeled.csv"
    all_metrics.to_csv(labeled_csv, index=False)
    print(f"✅ Saved labeled metrics: {labeled_csv}")

    # Confidence (over all episodes; sorted by window_id within-episode already)
    conf_df = compute_confidence(all_metrics, apm_low=args.apm_low, apm_high=args.apm_high, gap_high_ms=args.gap_high)
    conf_csv = outdir / "teamx_phase_confidence.csv"
    conf_df.to_csv(conf_csv, index=False)
    print(f"✅ Saved phase confidence: {conf_csv}")

    # Plots
    lines_png = outdir / "teamx_phase_confidence_lines.png"
    bar_png   = outdir / "teamx_phase_confidence_timeline.png"
    plot_confidence_lines(conf_df, lines_png)
    plot_top_phase_timeline(conf_df, bar_png)
    print(f"🖼  Saved charts:\n  - {lines_png}\n  - {bar_png}")

    # Persist pruned links bundle next to input
    linked_path = input_path.with_name(input_path.stem + "_linked.json")
    json_dump_pretty(linked_bundle, linked_path)
    print(f"🧩 Saved linked snapshot: {linked_path}")

    print("Done.")

if __name__ == "__main__":
    main()

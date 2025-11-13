from __future__ import annotations
import json, re, sys, os, time, io
import math
import requests
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# -----------------------------
# Config
# -----------------------------
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If a move has both text and image, fuse them.
# alpha = weight on text (0..1). 0.6 means 60% text, 40% image.
FUSE_TEXT_WEIGHT = 0.6

# Timeout for image downloads (seconds)
IMG_TIMEOUT = 15

# Some keys your Firestore moves may use
TEXT_KEYS = ("text", "label", "content")   # check these for text
URL_KEYS  = ("url", "imageUrl", "src")     # check these for image URL
TYPE_KEYS = ("shapeType", "type", "kind")  # where "image" or "text" may appear

# -----------------------------
# Helpers
# -----------------------------
def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize row-wise."""
    eps = 1e-8
    norms = torch.clamp(x.norm(dim=1, keepdim=True), min=eps)
    return x / norms

def _get_first_nonempty(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _is_image_move(m: Dict[str, Any]) -> bool:
    # Prefer explicit type if present; otherwise infer from URL presence
    t = _get_first_nonempty(m, TYPE_KEYS)
    if t and t.lower() == "image":
        return True
    url = _get_first_nonempty(m, URL_KEYS)
    return bool(url)

def _download_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=IMG_TIMEOUT)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

# -----------------------------
# CLIP embedder
# -----------------------------
class CLIPEmbedder:
    def __init__(self, model_id: str = CLIP_MODEL_ID, device: str = DEVICE):
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.to(device)
        self.device = device

    @torch.inference_mode()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.model.config.projection_dim))
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        feats = self.model.get_text_features(**inputs)
        return _normalize_rows(feats)

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, self.model.config.projection_dim))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return _normalize_rows(feats)

# -----------------------------
# Multimodal embedding for moves
# -----------------------------
def embed_moves(moves: List[Dict[str, Any]], clipper: CLIPEmbedder) -> torch.Tensor:
    """
    Returns an NxD tensor of normalized embeddings, one per move.
    Strategy:
      - If move has image only -> image embedding
      - If text only -> text embedding
      - If both -> weighted mean (FUSE_TEXT_WEIGHT * text + (1-alpha) * image), renormalized
      - If neither -> zero vector (will produce 0 sims)
    """
    N = len(moves)
    D = clipper.model.config.projection_dim
    out = torch.zeros((N, D), device=clipper.device)

    # Prepare batches
    text_indices, text_payloads = [], []
    image_indices, image_payloads = [], []

    # First pass: gather payloads
    for i, m in enumerate(moves):
        text = _get_first_nonempty(m, TEXT_KEYS)
        url  = _get_first_nonempty(m, URL_KEYS)

        if text:
            verbs = ["add_note", "edit", "move", "cluster", "highlight", "link", "group", "draw"]
            # also remove underscores and extra spaces
            text = re.sub(r"\b(" + "|".join(verbs) + r")\b", "", text, flags=re.IGNORECASE)
            text = re.sub(r"[_]+", " ", text)
            text = re.sub(r"\s+", " ", text).strip()


        has_text = bool(text)
        has_img  = bool(url)

        # If both present, we'll compute both and fuse later
        if has_text:
            text_indices.append(i)
            text_payloads.append(text)

        if has_img:
            img = _download_image(url)
            if img is not None:
                image_indices.append(i)
                image_payloads.append(img)

    # Batch encode
    text_embs  = clipper.encode_text(text_payloads)   if text_payloads  else torch.empty((0, D), device=clipper.device)
    image_embs = clipper.encode_images(image_payloads) if image_payloads else torch.empty((0, D), device=clipper.device)

    # Place into result buffers
    # We need quick lookup from move idx -> row in batch embedding
    tpos = {idx: k for k, idx in enumerate(text_indices)}
    ipos = {idx: k for k, idx in enumerate(image_indices)}

    for i in range(N):
        has_t = i in tpos
        has_i = i in ipos

        if has_t and has_i:
            tvec = text_embs[tpos[i]]
            ivec = image_embs[ipos[i]]
            fused = FUSE_TEXT_WEIGHT * tvec + (1.0 - FUSE_TEXT_WEIGHT) * ivec
            fused = fused / (fused.norm() + 1e-8)
            out[i] = fused
        elif has_t:
            out[i] = text_embs[tpos[i]]
        elif has_i:
            out[i] = image_embs[ipos[i]]
        else:
            # leave as zero vector (produces 0 similarity)
            pass

    return out

# -----------------------------
# Link computation
# -----------------------------
def compute_links_from_embs(embs: torch.Tensor) -> Dict[int, Dict[int, float]]:
    """
    Given NxD normalized embeddings (on DEVICE), compute lower-triangular
    cosine sims: links[i][j] = sim(i, j) for j < i
    """
    N = embs.shape[0]
    links: Dict[int, Dict[int, float]] = {i: {} for i in range(N)}
    if N == 0:
        return links

    # Cosine for normalized vectors is just dot product
    # Compute in blocks if you expect very large N
    for i in range(N):
        vi = embs[i]
        # skip all-zero rows (no modality)
        if torch.allclose(vi, torch.zeros_like(vi)):
            continue
        sims = (embs[:i] @ vi)  # vector of size i
        for j in range(i):
            links[i][j] = float(sims[j].item())
    return links

# -----------------------------
# I/O driver
# -----------------------------
def add_links_to_file(fpath: str):
    with open(fpath, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    clipper = CLIPEmbedder()

    out = {}
    for ep_id, episode in data.items():
        # 'episode' is expected to be a list[move]; each move is a dict that may include:
        #   text, shapeType, url, actor, timestamp, etc.
        print(f"Processing episode: {ep_id}  (moves={len(episode)})")

        # Embed all moves (text and/or image)
        embs = embed_moves(episode, clipper)

        # Compute links via cosine (dot for normalized)
        links = compute_links_from_embs(embs)

        # Save (optionally include a per-move modality tag for debugging)
        modalities = []
        for m in episode:
            t = _get_first_nonempty(m, TEXT_KEYS)
            u = _get_first_nonempty(m, URL_KEYS)
            if t and u:
                modalities.append("text+image")
            elif u:
                modalities.append("image")
            elif t:
                modalities.append("text")
            else:
                modalities.append("none")

        out[ep_id] = {
            "moves": episode,
            "links": links,
            "modality": modalities,   # helps you debug or visualize later
            "model": CLIP_MODEL_ID,
        }

    outpath = re.sub(r"\.json$", "_linked.json", fpath)
    if outpath == fpath:
        root, ext = os.path.splitext(fpath)
        outpath = f"{root}_linked{ext or '.json'}"

    with open(outpath, "w", encoding="utf-8") as outfile:
        json.dump(out, outfile, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {outpath}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    # Usage: python compute_links_multimodal.py /path/to/teamx_session_10min.json
    if len(sys.argv) < 2:
        print("Usage: python compute_links_multimodal.py <input.json>")
        sys.exit(1)
    start = time.time()
    add_links_to_file(sys.argv[1])
    print(f"Time elapsed: {time.time()-start:.2f}s")






import io, json, os
import re
import sys
import time
from PIL import Image
import requests
from typing import Dict, Any, Tuple, Optional, List
import torch
from transformers import CLIPModel, CLIPProcessor

IMG_TIMEOUT = 15
FUSE_TEXT_WEIGHT = 0.6
TYPE_KEYS = ("shapeType", "type", "kind")
TEXT_KEYS = ("text", "label", "content")
URL_KEYS  = ("url", "imageUrl","imageUrls", "src")
COMMAND_VERBS = {"add_note", "edit", "move", "cluster", "highlight", "link", "group", "draw"}

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------
# Helpers
# --------------

def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    norms = torch.clamp(x.norm(dim=1, keepdim=True), min=eps)
    return x / norms

def _get_first_nonempty(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
            return v[0].strip()
    return None

def _get_text_from_move(m: Dict[str, Any]) -> Optional[str]:
    # 1) top-level text/label/content
    txt = _get_first_nonempty(m, TEXT_KEYS)
    if txt:
        return txt
    # 2) nested in content
    content = m.get("content") or {}
    txt = _get_first_nonempty(content, TEXT_KEYS)
    return txt

def _get_image_url_from_move(m: Dict[str, Any]) -> Optional[str]:
    # 1) top-level url/imageUrl/imageUrls/src
    url = _get_first_nonempty(m, URL_KEYS)
    if url:
        return url
    # 2) nested in content
    content = m.get("content") or {}
    url = _get_first_nonempty(content, URL_KEYS)
    return url


def _is_image_move(m: Dict[str, Any]) -> bool:
    t = _get_first_nonempty(m, TYPE_KEYS)
    if t and t.lower() == "image":
        return True
    url = _get_first_nonempty(m, URL_KEYS)
    return bool(url)

# download image by URL
def _download_image(url: str) -> Optional[Image.Image]:
    try:
        # DEBUG: short log for attempted download
        print(f"[IMG] Trying to download: {url[:80]}...")
        r = requests.get(url, timeout=IMG_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img = img.convert("RGB")
        # DEBUG: success
        print(f"[IMG] Success: {url[:80]}  size={img.size}")
        return img
    except Exception as e:
        # DEBUG: failure
        print(f"[IMG] FAILED: {url[:80]}  error={e}")
        return None
    
# -------------------------
# CLIP EMBEDDER
# -------------------------

class CLIPEmbedder:
    def __init__(self, model_id: str = CLIP_MODEL_ID, device: str = DEVICE):
        print(f"[CLIP] Loading model {model_id} on {device}...")
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.to(device)
        self.device = device
        print(f"[CLIP] Model loaded.")

    @torch.inference_mode()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self.model.config.projection_dim), device=self.device)
        print(f"[CLIP] Encoding {len(texts)} text items...")
        inputs = self.processor(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return _normalize_rows(feats)

    @torch.inference_mode()
    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.empty((0, self.model.config.projection_dim), device=self.device)
        print(f"[CLIP] Encoding {len(images)} images...")
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return _normalize_rows(feats)
    
# ------------------------
# Multimodal embedding for moves
# ------------------------
def embed_moves(moves: List[Dict[str, Any]], clipper: CLIPEmbedder) -> torch.Tensor:
    """
    Returns an NxD tensor of normalized embeddings, one per move.
    Strategy:
        - If move has image only -> image embedding
        - If text only -> text embedding
        - If both -> fused embedding (text + image)
        - If neither -> zero vector (will produce 0 sim)    
    """
    N = len(moves)
    D = clipper.model.config.projection_dim
    out = torch.zeros((N, D), device=clipper.device)

    text_indices, text_payloads = [], []
    image_indices, image_payloads = [], []

    # DEBUG counters
    text_only = 0
    image_only = 0
    text_and_image = 0
    neither = 0

    for i, m in enumerate(moves):
        # text = _get_first_nonempty(m, TEXT_KEYS)
        # url = _get_first_nonempty(m, URL_KEYS)

        text = _get_text_from_move(m)
        url = _get_image_url_from_move(m)

        has_text = bool(text)
        has_img = bool(url)

        if has_text and has_img:
            text_and_image += 1
        elif has_text:
            text_only += 1
        elif has_img:
            image_only += 1
        else:
            neither += 1

        if has_text:
            text_indices.append(i)
            text_payloads.append(text)

        if has_img:
            img = _download_image(url)
            if img is not None:
                image_indices.append(i)
                image_payloads.append(img)
            else:
                # DEBUG: mark failed image
                print(f"[EMBED] Image download failed for move {i}, url={url[:80]}")

    # DEBUG: print modality breakdown for this episode
    print(f"[EMBED] Moves: {N} | text_only={text_only}, image_only={image_only}, "
          f"text+image={text_and_image}, neither={neither}")
    print(f"[EMBED] Unique text_indices={len(text_indices)}, image_indices={len(image_indices)}")

    # Batch encode
    text_embs = clipper.encode_text(text_payloads) if text_payloads else torch.empty((0, D), device=clipper.device)
    image_embs = clipper.encode_image(image_payloads) if image_payloads else torch.empty((0, D), device=clipper.device)

    tpos = {idx: k for k, idx in enumerate(text_indices)}
    ipos = {idx: k for k, idx in enumerate(image_indices)}

    for i in range(N):
        has_t = i in tpos
        has_i = i in ipos

        if has_i and has_t:
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
            # stays zero
            pass

    # DEBUG: check how many zero embeddings we still have
    norms = out.norm(dim=1)
    zero_vecs = (norms < 1e-6).sum().item()
    print(f"[EMBED] Zero-vector embeddings: {zero_vecs} / {N}")

    return out



# ----------------------
# LINK COMPUTATION
# ----------------------

def compute_links_from_embs(embs: torch.Tensor) -> Dict[int, Dict[int, float]]:
    """
    Given NxD normalized embeddings (on DEVICE), compute lower_triangular
    cosine sims: links[i][j] = sim(i,j) for j<i
    """

    N = embs.shape[0]
    links: Dict[int, Dict[int, float]] = {i: {} for i in range(N)}

    if N == 0:
        return links
    
    all_sims = []  # DEBUG: track for summary

    for i in range(N):
        vi = embs[i]
        if torch.allclose(vi, torch.zeros_like(vi)):
            # DEBUG: note skipped row
            print(f"[LINK] Skipping move {i} due to zero embedding")
            continue
        sims = (embs[:i] @ vi)
        if i > 0:
            all_sims.append(sims.detach().cpu())

        for j in range(i):
            links[i][j] = float(sims[j].item())
    
    # DEBUG: similarity distribution summary
    if all_sims:
        all_sims_tensor = torch.cat(all_sims)
        min_sim = float(all_sims_tensor.min().item())
        max_sim = float(all_sims_tensor.max().item())
        mean_sim = float(all_sims_tensor.mean().item())
        num_pos = int((all_sims_tensor > 0).sum().item())
        num_zero_or_neg = int((all_sims_tensor <= 0).sum().item())
        print(f"[LINK] Similarity stats: min={min_sim:.3f}, max={max_sim:.3f}, "
              f"mean={mean_sim:.3f}, >0={num_pos}, <=0={num_zero_or_neg}")
    else:
        print("[LINK] No similarities computed (all zero embeddings or N<=1).")

    return links

# ---------------
# I/O DRIVER
# ---------------

def add_links_to_file(fpath: str):
    print(f"[IO] Loading input file: {fpath}")
    with open(fpath, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    
    clipper = CLIPEmbedder()

    out = {}

    # If file is a single episode (list), normalize it to dict form
    if isinstance(data, list):
        data = {"episode_0": data}

    for ep_id, episode in data.items():
        print(f"\n---- Processing episode: {ep_id} (moves={len(episode)})")
        embs = embed_moves(episode, clipper)
        links = compute_links_from_embs(embs)

        # DEBUG: link-value summary
        all_vals = []
        for d in links.values():
            all_vals.extend(d.values())
        if all_vals:
            min_v = min(all_vals)
            max_v = max(all_vals)
            pos = sum(1 for v in all_vals if v > 0)
            zero = sum(1 for v in all_vals if v == 0)
            neg = sum(1 for v in all_vals if v < 0)
            print(f"[EP {ep_id}] Links: total={len(all_vals)}, min={min_v:.3f}, "
                  f"max={max_v:.3f}, >0={pos}, ==0={zero}, <0={neg}")
        else:
            print(f"[EP {ep_id}] No link values computed.")

        modalities = []
        for m in episode:
            t = _get_first_nonempty(m, TEXT_KEYS)
            u = _get_first_nonempty(m, URL_KEYS)

            if t and u:
                modalities.append("text_image")
            elif t:
                modalities.append("text")
            elif u:
                modalities.append("image")
            else:
                modalities.append("others")
        
        # DEBUG: modality summary
        from collections import Counter
        counts = Counter(modalities)
        print(f"[EP {ep_id}] Modality counts: {dict(counts)}")
            
        out[ep_id] = {
            "moves": episode,
            "links": links,
            "modality": modalities,
            "model": CLIP_MODEL_ID,
        }

    outpath = re.sub(r"\.json$", "_linked.json", fpath)
    if outpath == fpath:
        root, ext = os.path.splitext(fpath)
        outpath = f"{root}_linked{ext or '.json'}"

    with open(outpath, "w", encoding="utf-8") as outfile:
        json.dump(out, outfile, ensure_ascii=False, indent=2)
    
    print(f"\n[IO] --- DONE -- WROTE {outpath}")


# --------------------
# CLI INTERFACE
# --------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_links_multimodal_copy.py <input.json>")
        sys.exit(1)
    start = time.time()
    add_links_to_file(sys.argv[1])
    print(f"[IO] Time elapsed: {time.time() - start:.2f}s")

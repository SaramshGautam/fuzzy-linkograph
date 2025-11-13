# from __future__ import annotations
# import json, re, sys, os, time
# from typing import List, Dict, Optional, Tuple
# import torch
# from PIL import Image
# from transformers import CLIPModel, CLIPProcessor

# MODEL_ID = "openai/clip-vit-base-patch32"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 32  # adjust for your memory

# def _load_image(p: str) -> Image.Image:
#     img = Image.open(p).convert("RGB")
#     return img

# def _move_image_path(move: Dict) -> Optional[str]:
#     for k in ("image_path", "image", "path", "filepath"):
#         v = move.get(k)
#         if isinstance(v, str) and v.strip():
#             return v
#     return None  # <-- no error, just skip later

# def encode_images(image_paths: List[str], model: CLIPModel, processor: CLIPProcessor) -> torch.Tensor:
#     """Return L2-normalized image embeddings as a (N, D) tensor on CPU."""
#     feats = []
#     model.eval()
#     with torch.no_grad():
#         for i in range(0, len(image_paths), BATCH_SIZE):
#             batch_paths = image_paths[i : i + BATCH_SIZE]
#             batch_imgs = [_load_image(p) for p in batch_paths]
#             inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(DEVICE)
#             img_feats = model.get_image_features(**inputs)  # (B, D)
#             img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # cosine-ready
#             feats.append(img_feats.detach().cpu())
#     return torch.cat(feats, dim=0) if feats else torch.empty((0, 512))

# def compute_links_from_images(
#     index_path_pairs: List[Tuple[int, str]],
#     model: CLIPModel,
#     processor: CLIPProcessor
# ) -> Dict[int, Dict[int, float]]:
#     """
#     index_path_pairs: list of (original_index, image_path) ONLY for moves that have an image
#     Returns links dict with original indices: links[i][j] for i>j where both had images.
#     """
#     if len(index_path_pairs) < 2:
#         # not enough images to link; return empty structure for caller to merge
#         return {}

#     orig_indices = [i for i, _ in index_path_pairs]
#     paths = [p for _, p in index_path_pairs]

#     # Optional: warn on missing files but continue (skip those)
#     present_pairs = [(i, p) for i, p in index_path_pairs if os.path.exists(p)]
#     if len(present_pairs) < len(index_path_pairs):
#         missing = [p for _, p in index_path_pairs if not os.path.exists(p)]
#         print(f"Warning: {len(missing)} image file(s) not found. First missing: {missing[0]}")
#         if len(present_pairs) < 2:
#             return {}

#         orig_indices = [i for i, _ in present_pairs]
#         paths = [p for _, p in present_pairs]

#     embs = encode_images(paths, model, processor)  # (N, D)
#     sim = embs @ embs.T  # cosine (embs are unit-length)

#     # Build links keyed by original indices i>j
#     links: Dict[int, Dict[int, float]] = {}
#     for a in range(len(orig_indices)):
#         i = orig_indices[a]
#         for b in range(a):
#             j = orig_indices[b]
#             links.setdefault(i, {})[j] = float(sim[a, b].item())
#     return links

# def add_image_links_to_file(fpath: str):
#     with open(fpath, "r", encoding="utf-8") as infile:
#         data = json.load(infile)

#     print("Loading CLIP…")
#     model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
#     processor = CLIPProcessor.from_pretrained(MODEL_ID)
#     print(f"Model loaded on {DEVICE}")

#     out = {}
#     for ep_id, episode in data.items():
#         print(f"Processing episode: {ep_id}  (moves={len(episode)})")

#         # Gather only moves that have an image path
#         index_path_pairs: List[Tuple[int, str]] = []
#         for idx, move in enumerate(episode):
#             p = _move_image_path(move)
#             if p:  # keep only those with images
#                 index_path_pairs.append((idx, p))

#         # Initialize empty links for every move index so structure mirrors input
#         episode_links: Dict[int, Dict[int, float]] = {i: {} for i in range(len(episode))}

#         # Compute links among the subset that have images
#         sub_links = compute_links_from_images(index_path_pairs, model, processor)
#         # Merge back into full index space
#         for i, row in sub_links.items():
#             episode_links.setdefault(i, {}).update(row)

#         out[ep_id] = {
#             "moves": episode,
#             "links": episode_links,
#             "meta": {
#                 "model": MODEL_ID,
#                 "device": DEVICE,
#                 "moves_with_images": len(index_path_pairs),
#                 "total_moves": len(episode),
#             },
#         }

#     outpath = re.sub(r"\.json$", "_linked_images.json", fpath)
#     if outpath == fpath:
#         root, ext = os.path.splitext(fpath)
#         outpath = f"{root}_linked_images{ext or '.json'}"

#     with open(outpath, "w", encoding="utf-8") as outfile:
#         json.dump(out, outfile, ensure_ascii=False, indent=2)
#     print(f"✅ Wrote {outpath}")

# if __name__ == "__main__":
#     # Usage: python compute_links_clip.py /path/to/input.json
#     if len(sys.argv) < 2:
#         print("Usage: python compute_links_clip.py <input.json>")
#         sys.exit(1)
#     start = time.time()
#     add_image_links_to_file(sys.argv[1])
#     print(f"Time elapsed: {time.time()-start:.2f}s")


from __future__ import annotations
import json, re, sys, os, time
from io import BytesIO
from typing import Dict, List, Optional
import torch
from PIL import Image
import requests
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMEOUT = 15  # seconds

def _extract_image_sources(move: Dict) -> List[str]:
    """Return a list of image sources (paths or URLs) for a move."""
    srcs: List[str] = []
    # common top-level keys
    for k in ("image_path", "image", "path", "filepath", "image_url"):
        v = move.get(k)
        if isinstance(v, str) and v.strip():
            srcs.append(v.strip())

    # array variants at top-level
    for k in ("images", "image_paths", "image_urls"):
        v = move.get(k)
        if isinstance(v, list):
            srcs.extend([s for s in v if isinstance(s, str) and s.strip()])

    # nested under content
    content = move.get("content", {})
    v1 = content.get("imageUrls") or content.get("image_urls") or content.get("images")
    if isinstance(v1, list):
        srcs.extend([s for s in v1 if isinstance(s, str) and s.strip()])
    v2 = content.get("image") or content.get("image_path") or content.get("image_url")
    if isinstance(v2, str) and v2.strip():
        srcs.append(v2.strip())

    # de-dup & preserve order
    seen = set()
    out = []
    for s in srcs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _load_image_any(src: str) -> Optional[Image.Image]:
    """Load a PIL image from a local path or HTTP(S) URL. Returns None on failure."""
    try:
        if src.startswith("http://") or src.startswith("https://"):
            r = requests.get(src, timeout=TIMEOUT)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
        else:
            if not os.path.exists(src):
                return None
            img = Image.open(src).convert("RGB")
        return img
    except Exception:
        return None

def _move_embedding(
    imgs: List[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor
) -> Optional[torch.Tensor]:
    """Average-normalized CLIP image embedding for a list of PIL images."""
    if not imgs:
        return None
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
        feats = model.get_image_features(**inputs)  # (B, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # unit vectors
        mean = feats.mean(dim=0, keepdim=True)           # (1, D)
        mean = mean / mean.norm(dim=-1, keepdim=True)    # re-normalize
        return mean.squeeze(0).cpu()                     # (D,)

def compute_links_for_episode(episode: List[Dict], model: CLIPModel, processor: CLIPProcessor) -> Dict[int, Dict[int, float]]:
    # Build per-move embedding (average over all images in that move)
    move_vecs: Dict[int, torch.Tensor] = {}
    for i, move in enumerate(episode):
        sources = _extract_image_sources(move)
        imgs = [im for src in sources if (im := _load_image_any(src)) is not None]
        vec = _move_embedding(imgs, model, processor)
        if vec is not None:
            move_vecs[i] = vec

    # Compute cosine similarity for pairs where both moves have embeddings
    links: Dict[int, Dict[int, float]] = {i: {} for i in range(len(episode))}
    keys = sorted(move_vecs.keys())
    if len(keys) < 2:
        return links

    # Stack for speed
    vec_mat = torch.stack([move_vecs[k] for k in keys], dim=0)   # (N, D)
    vec_mat = vec_mat / vec_mat.norm(dim=-1, keepdim=True)       # ensure unit
    sim = vec_mat @ vec_mat.T                                    # (N, N) cosine

    for a, i in enumerate(keys):
        for b in range(a):
            j = keys[b]
            links[i][j] = float(sim[a, b].item())
    return links

def add_image_links_to_file(fpath: str):
    with open(fpath, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    print("Loading CLIP…")
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    print(f"Model loaded on {DEVICE}")

    out = {}
    for ep_id, episode in data.items():
        print(f"Processing episode: {ep_id}  (moves={len(episode)})")
        links = compute_links_for_episode(episode, model, processor)

        # count moves with at least one usable image
        moves_with_images = sum(1 for i in range(len(episode)) if links.get(i) or any(i in row for row in links.values()))

        out[ep_id] = {
            "moves": episode,
            "links": links,
            "meta": {
                "model": MODEL_ID,
                "device": DEVICE,
                "moves_with_images": moves_with_images,
                "total_moves": len(episode),
            },
        }

    outpath = re.sub(r"\.json$", "_linked_images.json", fpath)
    if outpath == fpath:
        root, ext = os.path.splitext(fpath)
        outpath = f"{root}_linked_images{ext or '.json'}"

    with open(outpath, "w", encoding="utf-8") as outfile:
        json.dump(out, outfile, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {outpath}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_links_clip.py <input.json>")
        sys.exit(1)
    start = time.time()
    add_image_links_to_file(sys.argv[1])
    print(f"Time elapsed: {time.time()-start:.2f}s")

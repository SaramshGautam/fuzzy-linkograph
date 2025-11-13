from __future__ import annotations
from sentence_transformers import SentenceTransformer, util
import json, re, sys, os, time

MODEL_ID = "all-MiniLM-L6-v2"

def compute_links(moves_text, model):
    # Batch-encode for speed; normalize embeddings for cosine
    embs = model.encode(moves_text, normalize_embeddings=True)
    links = {}
    for i in range(len(moves_text)):
        links[i] = {}
        for j in range(i):
            # util.cos_sim expects tensors/arrays; returns 1x1 tensor here
            sim = float(util.cos_sim(embs[i], embs[j]).item())
            links[i][j] = sim
    return links

def add_links_to_file(fpath: str):
    with open(fpath, "r", encoding="utf-8") as infile:
        data = json.load(infile)

    # Load model once
    model = SentenceTransformer(MODEL_ID)

    out = {}
    for ep_id, episode in data.items():
        print(f"Processing episode: {ep_id}  (moves={len(episode)})")
        moves_text = [m.get("text","") for m in episode]
        out[ep_id] = {
            "moves": episode,
            "links": compute_links(moves_text, model),
        }

    outpath = re.sub(r"\.json$", "_linked.json", fpath)
    if outpath == fpath:  # in case extension wasn’t .json
        root, ext = os.path.splitext(fpath)
        outpath = f"{root}_linked{ext or '.json'}"

    with open(outpath, "w", encoding="utf-8") as outfile:
        json.dump(out, outfile, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {outpath}")

if __name__ == "__main__":
    # Usage: python compute_links_simple.py /path/to/teamx_session_10min.json
    if len(sys.argv) < 2:
        print("Usage: python compute_links_simple.py <input.json>")
        sys.exit(1)
    start = time.time()
    add_links_to_file(sys.argv[1])
    print(f"Time elapsed: {time.time()-start:.2f}s")

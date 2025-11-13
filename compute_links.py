from sentence_transformers import SentenceTransformer, util
import json
import re
import time

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_links(moves):
	embs = [model.encode(move) for move in moves]
	links = {}
	for i in range(len(moves)):
		links[i] = {}
		for j in range(i):
			links[i][j] = util.cos_sim([embs[i]], [embs[j]])[0].item()
	return links

def add_links_to_file(fpath):
	with open(fpath, "r") as infile:
		data = json.load(infile)
		for ep_id, episode in data.items():
			print(ep_id)
			data[ep_id] = {
				"moves": episode,
				"links": compute_links([move["text"] for move in episode])
			}
		with open(re.sub(r"\.json$", "_linked.json", fpath), "w") as outfile:
			json.dump(data, outfile)

start = time.time()
add_links_to_file("data/TeamD_formative.json")
end = time.time()
print("Time elapsed:", end - start)

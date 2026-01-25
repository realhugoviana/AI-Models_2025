import torch
import torch.nn.functional as F
import pickle

with open("gui/celebrity_centroids.pkl", "rb") as f:
    celebrity_centroids = pickle.load(f)

def closest_celebrity(emb, celeb_dict):
    best_name = None
    best_sim = -1
    best_sex = None

    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().float().view(1, -1)

    for name, data in celeb_dict.items():

        # 🔎 Extraction robuste de l'embedding
        if isinstance(data, dict):
            centroid = data.get("embedding")
            sex = data.get("sex")
        elif isinstance(data, (list, tuple)):
            centroid = data[0]
            sex = data[1] if len(data) > 1 else None
        else:
            centroid = data
            sex = None

        if centroid is None:
            continue

        # 🔄 Conversion numpy → torch
        if not isinstance(centroid, torch.Tensor):
            centroid = torch.tensor(centroid)

        centroid = centroid.float().view(1, -1)

        # 🔢 Similarité cosinus
        sim = F.cosine_similarity(emb, centroid).item()

        if sim > best_sim:
            best_sim = sim
            best_name = name
            best_sex = sex

    return best_name, best_sim, best_sex
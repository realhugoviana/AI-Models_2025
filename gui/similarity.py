import pickle
import torch
from torch.nn.functional import cosine_similarity

with open("celebrity_centroids.pkl", "rb") as f:
    celebrity_data = pickle.load(f)



def closest_celebrity(face_embedding, celeb_dict):
    best_name = None
    best_sim = -1
    best_sex = None
    for name, data in celeb_dict.items():
        sim = cosine_similarity(face_embedding, data["embedding"].unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_name = name
            best_sex = data["sex"]
    return best_name, best_sim, best_sex

import os
import pickle
import torch
import numpy as np
from PIL import Image
from retinaface import RetinaFace
from torchvision import transforms

from src.multi_head_classifier.multi_head_model import MultiHeadClassifier

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = "Dataset/train"
OUTPUT_FILE = "gui/celebrity_centroids.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load model (embedding only)
# --------------------------------------------------

model = MultiHeadClassifier.load_from_checkpoint(
    "models/multi_head_classifier.ckpt",
    map_location=DEVICE
)

embedding_model = model.embedding_model.to(DEVICE).eval()
torch.set_grad_enabled(False)

# --------------------------------------------------
# Image transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def extract_largest_face(image: Image.Image):
    img = np.array(image)[:, :, ::-1]
    faces = RetinaFace.detect_faces(img)

    if faces is None:
        return None

    best = max(
        faces.values(),
        key=lambda f: (f["facial_area"][2] - f["facial_area"][0]) *
                      (f["facial_area"][3] - f["facial_area"][1])
    )

    x1, y1, x2, y2 = best["facial_area"]
    face = img[y1:y2, x1:x2]
    return Image.fromarray(face[:, :, ::-1])


def compute_embedding(face_img: Image.Image):
    x = transform(face_img).unsqueeze(0).to(DEVICE)
    emb = embedding_model(x)
    return emb.squeeze(0).cpu()

# --------------------------------------------------
# Main loop
# --------------------------------------------------

centroids = {}

for celeb in sorted(os.listdir(DATASET_DIR)):
    celeb_dir = os.path.join(DATASET_DIR, celeb)

    if not os.path.isdir(celeb_dir):
        continue

    print(f"Processing {celeb}...")
    embeddings = []

    for fname in os.listdir(celeb_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(celeb_dir, fname)

        try:
            image = Image.open(img_path).convert("RGB")
            face = extract_largest_face(image)
            if face is None:
                continue

            emb = compute_embedding(face)
            embeddings.append(emb)

        except Exception as e:
            print(f"  Skipped {fname}: {e}")

    if len(embeddings) == 0:
        print(f"  ⚠️ No valid faces for {celeb}")
        continue

    centroid = torch.stack(embeddings).mean(dim=0)
    centroids[celeb] = centroid

    print(f"  ✔ {len(embeddings)} faces used")

# --------------------------------------------------
# Save
# --------------------------------------------------

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(centroids, f)

print(f"\nSaved {len(centroids)} celebrity centroids to {OUTPUT_FILE}")

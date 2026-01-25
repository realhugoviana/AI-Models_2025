import json
from gui.detectors import contains_person, extract_face
from gui.embeddings import get_embedding
from gui.classifier import classify
from gui.similarity import closest_celebrity, celebrity_centroids
import os

THRESHOLD = 0.5

def predict_pipeline(image, want_name, want_sex):
    want_name = (want_name == "Yes")
    want_sex = (want_sex == "Yes")

    if not contains_person(image):
        return "❌ No person detected", None

    face = extract_face(image)
    if face is None:
        return "❌ No face detected", None

    emb = get_embedding(face)
    cls = classify(emb)

    print("Embedding mean:", emb.mean().item())
    print("Embedding std :", emb.std().item())
    print("Embedding norm:", emb.norm().item())
    celeb_name, sim, celeb_sex = closest_celebrity(emb, celebrity_centroids)

    result = {}

    if want_sex:
        result["Sex"] = str(cls["sex"])

    if want_name:
        if sim < THRESHOLD:
            result["Name"] = "Unknown"
            result["Closest celebrity"] = celeb_name
            result["Similarity"] = round(sim, 2)
            result["Celebrity sex"] = celeb_sex
        else:
            result["Name"] = celeb_name
            result["Confidence"] = round(cls["name_conf"], 2)
            result["Celebrity sex"] = celeb_sex

    # 7️⃣ Image célébrité (si dispo)
    celeb_img_path = f"gui/assets/celeb_images/{celeb_name}.jpg"

    if celeb_name is None or not os.path.exists(celeb_img_path):
        celeb_img_path = None  # Gradio accepte None

    return result, celeb_img_path
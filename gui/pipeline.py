from gui.detectors import contains_person, extract_face
from gui.embeddings import get_embedding
from gui.classifier import classify
from gui.similarity import closest_celebrity

THRESHOLD = 0.5

def predict_pipeline(image, want_name, want_sex):
    # 1️⃣ Check if image contains a person
    if not contains_person(image):
        return {"Error": "No person detected"}, None

    # 2️⃣ Extract face
    face = extract_face(image)
    if face is None:
        return {"Error": "No face detected"}, None

    # 3️⃣ Get embedding
    emb = get_embedding(face)

    # 4️⃣ Multi-head classification
    cls = classify(emb)

    # 5️⃣ Find closest celebrity
    celeb_name, sim, celeb_sex = closest_celebrity(emb)

    # 6️⃣ Prepare results dictionary
    result = {}

    if want_sex:
        result["Sex"] = cls["sex"]

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

    # 7️⃣ Return results and path to closest celebrity image
    return result, f"gui/assets/celeb_images/{celeb_name}.jpg"

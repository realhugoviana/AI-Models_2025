import numpy as np
from retinaface import RetinaFace
from gui.models import yolo

def contains_person(image):
    img = np.array(image)
    results = yolo(img, conf=0.4)[0]

    return any(int(c) == 0 for c in results.boxes.cls)

def extract_face(image):
    img = np.array(image)[:, :, ::-1]
    faces = RetinaFace.detect_faces(img)

    if faces is None:
        return None

    best = max(
        faces.values(),
        key=lambda f: (f["facial_area"][2]-f["facial_area"][0]) *
                      (f["facial_area"][3]-f["facial_area"][1])
    )

    x1, y1, x2, y2 = best["facial_area"]
    face = img[y1:y2, x1:x2]
    return face[:, :, ::-1]

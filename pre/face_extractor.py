import math
import numpy as np
import cv2
from PIL import Image
import dlib
from pathlib import Path
from retinaface import RetinaFace
import warnings
warnings.filterwarnings("ignore")

repo_path = Path("./working")
needle = "person"

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

for img_path in repo_path.rglob("*"):
    if not img_path.is_file():
        continue

    # ---- Delete images without label "person" ----
    if needle not in img_path.as_posix().lower():
        img_path.unlink()
        print(f"Deleted: {img_path}")
        continue

    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    if needle not in img_path.name:
        continue

    print(f"Processing: {img_path}")

    # ---- Read image (grayscale-safe) ----
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  -> unreadable, skipped")
        continue

    # ---- Convert grayscale to BGR ----
    if len(img.shape) == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:  # single channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # --- RetinaFace ---
    resp = RetinaFace.detect_faces(img)
    if not resp:
        img_path.unlink()
        print("  -> no face, deleted")
        continue

    face_key = list(resp.keys())[0]
    lm = resp[face_key]["landmarks"]

    x1, y1 = lm["right_eye"]
    x2, y2 = lm["left_eye"]

    a = abs(y1 - y2)
    b = abs(x2 - x1)

    if b == 0 or a == 0:
        continue

    alpha = math.degrees(math.atan2(a, b))
    eyes_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    M = cv2.getRotationMatrix2D(eyes_center, alpha, 1.0)

    aligned_img = cv2.warpAffine(
        img,
        M,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_CUBIC
    )

    # --- Dlib landmarks ---
    faces = face_detector(aligned_img, 1)
    if not faces:
        continue

    landmarks = landmark_detector(aligned_img, faces[0])

    landmarks_tuple = [
        (landmarks.part(i).x, landmarks.part(i).y) for i in range(68)
    ]

    routes = (
        list(range(16, -1, -1)) +
        list(range(17, 19)) +
        list(range(24, 26)) +
        [16]
    )

    routes_coordinates = [landmarks_tuple[i] for i in routes]

    mask = np.zeros(aligned_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(routes_coordinates), 1)

    out = np.zeros_like(aligned_img)
    out[mask.astype(bool)] = aligned_img[mask.astype(bool)]

    cv2.imwrite(str(img_path), out)

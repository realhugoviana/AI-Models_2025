import math
import numpy as np # type: ignore
import pandas as pd # type: ignore
import cv2 # type: ignore
from PIL import Image # type: ignore
import dlib # type: ignore
from pathlib import Path
from retinaface import RetinaFace # type: ignore
import time
import warnings
import os
warnings.filterwarnings("ignore")

repo_path = Path("../working")
needle = "person"

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)

# Save results for evaluation
results_df = pd.DataFrame({"celebrity": [], # Name of the celebrity on the image
                        "original_image": [], # Name of the original image
                        "yolo_bounding_box": [], # Name of the cropped yolo bounding box
                        "face_extracted": [], # Whether a face was extracted or not
                        "prediction_time": []}) # Prediction time of the image


# Access celebs subdirectory
for celeb in os.listdir(repo_path):
    celeb_path = Path(repo_path) / celeb
    celeb = celeb.replace(" ", "-")
    os.makedirs(f"../working_retinaface/{celeb}",  exist_ok=True)

    # Access images in the subdirectory
    for img_file in os.listdir(celeb_path):
        img_path = Path(celeb_path) / img_file
        bounding_box_name = img_file.replace(".jpg", "")
        bounding_box_name = bounding_box_name.replace(" ", "-")
        
        # Ignore non files
        if not img_path.is_file():
            print(f"Ignored : Not a file ({img_path})")
            continue

        # Ignore images without label "person"
        if needle not in img_path.as_posix().lower():
            print(f"Ignored: Not a person ({img_path})")
            continue
        
        # Ignore images that aren't jpg, jpeg or png
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            print(f"Ignored: Not a .jpg, .jpeg, .png ({img_path})")
            continue

        print(f"Processing: {img_path}")

        # get original image name
        original_img = bounding_box_name[:bounding_box_name.index("person")]

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

        # Measure prediction time
        start = time.time()

        # Extract face with RetinaFace 
        resp = RetinaFace.detect_faces(img)

        # Measure prediction time
        end = time.time()
        prediction_time = end - start

        # Save results in dataframe for evaluation
        r = pd.DataFrame({"celebrity": [celeb],
                        "original_image": [original_img],
                        "yolo_bounding_box": [bounding_box_name],
                        "face_extracted": [1], 
                        "prediction_time": [prediction_time]})

        if not resp:
            print("  -> no face extracted")
            r["face_extracted"] = 0
            results_df = pd.concat([results_df, r])
            continue

        face_key = list(resp.keys())[0]
        lm = resp[face_key]["landmarks"]

        x1, y1 = lm["right_eye"]
        x2, y2 = lm["left_eye"]

        a = abs(y1 - y2)
        b = abs(x2 - x1)

        if b == 0 or a == 0:
            print(" -> Eyes not detected")
            r["face_extracted"] = 0
            results_df = pd.concat([results_df, r])
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
            print(" -> face extracted but no face detected after alignment")
            r["face_extracted"] = 0
            results_df = pd.concat([results_df, r])
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


        output_path = Path("../working_retinaface") / celeb / f"{bounding_box_name}.jpg"
        cv2.imwrite(str(output_path), out)

        results_df = pd.concat([results_df, r])

# Save results to csv for evaluation
results_df.to_csv("pre/results/results_retina-face.csv", index=False)

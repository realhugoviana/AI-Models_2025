import math
import numpy as np 
import pandas as pd 
import cv2 
from PIL import Image
import dlib 
from pathlib import Path
from retinaface import RetinaFace
import time
import warnings
import os
import argparse
warnings.filterwarnings("ignore")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Face alignment and extraction pipeline')
parser.add_argument('--input_dir', type=str, default='../working',
                    help='Input directory containing celebrity subdirectories (default: ../working)')
parser.add_argument('--output_dir', type=str, default='../working_retinaface',
                    help='Output directory for aligned faces (default: ../working_retinaface)')
parser.add_argument('--max_rotation', type=float, default=10.0,
                    help='Maximum allowed rotation angle in degrees (default: 10.0)')
parser.add_argument('--min_confidence', type=float, default=0.95,
                    help='Minimum detection confidence score (default: 0.95)')
parser.add_argument('--min_eye_distance', type=float, default=30.0,
                    help='Minimum eye distance in pixels (default: 30.0)')
parser.add_argument('--max_eye_distance', type=float, default=200.0,
                    help='Maximum eye distance in pixels (default: 200.0)')
parser.add_argument('--landmark_model', type=str, default='shape_predictor_68_face_landmarks.dat',
                    help='Path to dlib landmark model (default: shape_predictor_68_face_landmarks.dat)')
parser.add_argument('--results_csv', type=str, default='pre/results/results_retina-face.csv',
                    help='Output CSV file for results (default: pre/results/results_retina-face.csv)')
parser.add_argument('--needle', type=str, default='person',
                    help='String to search for in filename (default: person)')

args = parser.parse_args()

repo_path = Path(args.input_dir)
needle = args.needle

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(args.landmark_model)

print(f"Configuration:")
print(f"  Input directory: {args.input_dir}")
print(f"  Output directory: {args.output_dir}")
print(f"  Max rotation: {args.max_rotation}°")
print(f"  Min confidence: {args.min_confidence}")
print(f"  Eye distance range: {args.min_eye_distance}-{args.max_eye_distance}px")
print(f"  Results CSV: {args.results_csv}")
print()

# Build results list for better performance
results_list = []

# Access celebs subdirectory
for celeb in os.listdir(repo_path):
    celeb_path = Path(repo_path) / celeb
    celeb = celeb.replace(" ", "-")
    os.makedirs(f"{args.output_dir}/{celeb}", exist_ok=True)

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

        # --- Dlib landmarks ---
        faces = face_detector(img, 1)
        if not faces:
            print(" -> no face detected by dlib")
            results_list.append({
                "celebrity": celeb,
                "original_image": original_img,
                "yolo_bounding_box": bounding_box_name,
                "face_extracted": 0,
                "prediction_time": 0
            })
            continue

        landmarks = landmark_detector(img, faces[0])

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

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(routes_coordinates), 1)

        masked_face = np.zeros_like(img)
        masked_face[mask.astype(bool)] = img[mask.astype(bool)]

        # Measure prediction time
        start = time.time()

        # Extract face with RetinaFace 
        resp = RetinaFace.detect_faces(masked_face)

        # Measure prediction time
        end = time.time()
        prediction_time = end - start

        if not resp:
            print("  -> no face extracted by RetinaFace")
            results_list.append({
                "celebrity": celeb,
                "original_image": original_img,
                "yolo_bounding_box": bounding_box_name,
                "face_extracted": 0,
                "prediction_time": prediction_time
            })
            continue

        face_key = list(resp.keys())[0]
        
        # Check detection confidence
        if resp[face_key]["score"] < args.min_confidence:
            print(f"  -> low confidence ({resp[face_key]['score']:.2f}), skipped")
            results_list.append({
                "celebrity": celeb,
                "original_image": original_img,
                "yolo_bounding_box": bounding_box_name,
                "face_extracted": 0,
                "prediction_time": prediction_time
            })
            continue
        
        lm = resp[face_key]["landmarks"]

        x1, y1 = lm["right_eye"]
        x2, y2 = lm["left_eye"]

        # Calculate deltas with sign
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            print(" -> Eyes vertically aligned, skipped")
            results_list.append({
                "celebrity": celeb,
                "original_image": original_img,
                "yolo_bounding_box": bounding_box_name,
                "face_extracted": 0,
                "prediction_time": prediction_time
            })
            continue

        # Calculate rotation angle
        alpha = math.degrees(math.atan2(dy, dx))
        
        # Check for extreme rotations
        if abs(alpha) > args.max_rotation:
            print(f"  -> Rotation too extreme ({alpha:.2f}°), skipped")
            results_list.append({
                "celebrity": celeb,
                "original_image": original_img,
                "yolo_bounding_box": bounding_box_name,
                "face_extracted": 0,
                "prediction_time": prediction_time
            })
            continue
        
        # Check for reasonable eye distance (filters profile views)
        eye_distance = math.sqrt(dx**2 + dy**2)
        if eye_distance < args.min_eye_distance or eye_distance > args.max_eye_distance:
            print(f"  -> Unusual eye distance ({eye_distance:.1f}px), skipped")
            results_list.append({
                "celebrity": celeb,
                "original_image": original_img,
                "yolo_bounding_box": bounding_box_name,
                "face_extracted": 0,
                "prediction_time": prediction_time
            })
            continue

        # Center based on facial landmarks
        h, w = masked_face.shape[:2]
        img_center = (w // 2, h // 2)
        
        # Get cheek landmarks for horizontal centering (landmarks 2 and 16)
        left_cheek = landmarks_tuple[2]   # Left side of face
        right_cheek = landmarks_tuple[16]  # Right side of face
        
        # Calculate horizontal center of the face
        face_center_x = (left_cheek[0] + right_cheek[0]) / 2
        
        # Get eyebrow landmarks (20-25 route) and chin (landmark 9) for vertical centering
        eyebrow_points = [landmarks_tuple[i] for i in [20, 25]]
        eyebrow_center_y = sum(p[1] for p in eyebrow_points) / len(eyebrow_points)
        chin_y = landmarks_tuple[9][1]
        
        # Calculate vertical center of the face (midpoint between eyebrow center and chin)
        face_center_y = (eyebrow_center_y + chin_y) / 2
        
        # Calculate translation to center the face
        tx = img_center[0] - face_center_x
        ty = img_center[1] - face_center_y
        
        # Create rotation matrix around image center
        M_rot = cv2.getRotationMatrix2D(img_center, alpha, 1.0)
        
        # Add translation to center the face based on landmarks
        M_rot[0, 2] += tx
        M_rot[1, 2] += ty

        # Apply combined transformation
        aligned_img = cv2.warpAffine(
            masked_face,
            M_rot,
            (w, h),
            flags=cv2.INTER_CUBIC
        )

        output_path = Path(args.output_dir) / celeb / f"{bounding_box_name}.jpg"
        cv2.imwrite(str(output_path), aligned_img)

        results_list.append({
            "celebrity": celeb,
            "original_image": original_img,
            "yolo_bounding_box": bounding_box_name,
            "face_extracted": 1,
            "prediction_time": prediction_time
        })

# Convert list to DataFrame once at the end (much faster)
results_df = pd.DataFrame(results_list)

# Save results to csv for evaluation
os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
results_df.to_csv(args.results_csv, index=False)

print(f"\nProcessing complete!")
print(f"Total images processed: {len(results_df)}")
print(f"Faces successfully extracted: {results_df['face_extracted'].sum()}")
print(f"Failed extractions: {len(results_df) - results_df['face_extracted'].sum()}")
print(f"Results saved to: {args.results_csv}")
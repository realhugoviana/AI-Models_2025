import pandas as pd
from pathlib import Path
import shutil

# ==============================
# Configuration
# ==============================
INPUT_CSV = "pre/results/results_bounding_boxes.csv"
WORKING_DIR = Path("../working")
OUTPUT_DIR = Path("analysis")
PERSON_CLASS = "person"

# ==============================
# Output directories
# ==============================
METRICS_DIR = OUTPUT_DIR / "metrics"
NON_PERSON_DIR = OUTPUT_DIR / "non_person_objects"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
NON_PERSON_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Load CSV
# ==============================
df = pd.read_csv(INPUT_CSV)

df["class"] = df["class"].astype(str)
df["image"] = df["image"].astype(str)
df["celebrity"] = df["celebrity"].astype(str)
df["confidence"] = (
    df["confidence"]
    .astype(str)
    .str.replace("tensor\\(\\[", "", regex=True)
    .str.replace("\\]\\)", "", regex=True)
    .astype(float)
)
df["prediction_time"] = df["prediction_time"].astype(float)

# ==============================
# Image-level aggregation
# ==============================
image_level = (
    df.groupby(["celebrity", "image"])
    .agg(
        person_detected=("class", lambda x: PERSON_CLASS in x.values),
        prediction_time=("prediction_time", "first")
    )
    .reset_index()
)

total_images = len(image_level)
images_with_person = image_level["person_detected"].sum()
reliability = images_with_person / total_images if total_images > 0 else 0.0

# ==============================
# Timing statistics
# ==============================
time_mean = image_level["prediction_time"].mean()
time_std = image_level["prediction_time"].std()

# ==============================
# Person confidence statistics
# ==============================
person_detections = df[df["class"] == PERSON_CLASS]

confidence_mean = person_detections["confidence"].mean()
confidence_std = person_detections["confidence"].std()

# ==============================
# Save metrics
# ==============================
reliability_df = pd.DataFrame({
    "total_images": [total_images],
    "images_with_person_detected": [images_with_person],
    "person_detection_reliability": [reliability]
})

reliability_df.to_csv(
    METRICS_DIR / "reliability_summary.csv",
    index=False
)

stats_df = pd.DataFrame({
    "prediction_time_mean": [time_mean],
    "prediction_time_std": [time_std],
    "person_confidence_mean": [confidence_mean],
    "person_confidence_std": [confidence_std]
})

stats_df.to_csv(
    METRICS_DIR / "timing_and_confidence_stats.csv",
    index=False
)

# ==============================
# Non-person detections
# ==============================
non_person_df = df[df["class"] != PERSON_CLASS].copy()

non_person_df.to_csv(
    OUTPUT_DIR / "non_person_detections.csv",
    index=False
)

# ==============================
# Organize cropped images
# ==============================
for _, row in non_person_df.iterrows():
    celeb = row["celebrity"]
    image = row["image"]
    obj_class = row["class"]

    class_dir = NON_PERSON_DIR / obj_class
    class_dir.mkdir(exist_ok=True)

    celeb_dir = WORKING_DIR / celeb
    if not celeb_dir.exists():
        continue

    for file in celeb_dir.glob(f"{image}-{obj_class}-nn-bb-*.jpg"):
        target = class_dir / file.name
        if not target.exists():
            shutil.copy(file, target)

print("Analysis complete.")

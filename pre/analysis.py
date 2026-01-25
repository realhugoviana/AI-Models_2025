import pandas as pd
from pathlib import Path

# ==============================
# Configuration
# ==============================
INPUT_CSV = "pre/results/results_retina-face.csv"
OUTPUT_DIR = Path("analysis")

METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Load CSV
# ==============================
df = pd.read_csv(INPUT_CSV)

df["celebrity"] = df["celebrity"].astype(str)
df["original_image"] = df["original_image"].astype(str)
df["face_extracted"] = df["face_extracted"].astype(int)
df["prediction_time"] = df["prediction_time"].astype(float)

# ==============================
# Image-level aggregation
# ==============================
image_level = (
    df.groupby(["celebrity", "original_image"])
    .agg(
        face_detected=("face_extracted", "max"),
        prediction_time=("prediction_time", "first"),
    )
    .reset_index()
)

total_images = len(image_level)
images_with_face = image_level["face_detected"].sum()
reliability = images_with_face / total_images if total_images > 0 else 0.0

# ==============================
# Timing statistics
# ==============================
time_mean = image_level["prediction_time"].mean()
time_std = image_level["prediction_time"].std()

# ==============================
# Save metrics
# ==============================
reliability_df = pd.DataFrame({
    "total_images": [total_images],
    "images_with_face_extracted": [images_with_face],
    "face_extraction_reliability": [reliability],
})

reliability_df.to_csv(
    METRICS_DIR / "retinaface_reliability_summary.csv",
    index=False
)

stats_df = pd.DataFrame({
    "prediction_time_mean": [time_mean],
    "prediction_time_std": [time_std],
})

stats_df.to_csv(
    METRICS_DIR / "retinaface_timing_stats.csv",
    index=False
)

print("RetinaFace analysis complete.")

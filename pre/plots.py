import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# Config
# ==============================
INPUT_CSV = "pre/results/results_bounding_boxes.csv"
OUTPUT_DIR = Path("analysis/plots")
PERSON_CLASS = "person"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# Load & clean
# ==============================
df = pd.read_csv(INPUT_CSV)

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
        prediction_time=("prediction_time", "first"),
        num_detections=("class", "count")
    )
    .reset_index()
)

# ==============================
# A. Reliability bar plot
# ==============================
reliability = image_level["person_detected"].mean()

plt.figure()
plt.bar(["Person detected"], [reliability])
plt.ylim(0, 1)
plt.ylabel("Proportion")
plt.title("Image-level person detection reliability")
plt.savefig(OUTPUT_DIR / "person_reliability.png")
plt.close()

# ==============================
# B. Prediction time histogram
# ==============================
plt.figure()
plt.hist(image_level["prediction_time"], bins=30)
plt.xlabel("Prediction time (s)")
plt.ylabel("Number of images")
plt.title("Prediction time distribution")
plt.savefig(OUTPUT_DIR / "prediction_time_hist.png")
plt.close()

# ==============================
# C. Person confidence histogram
# ==============================
person_conf = df[df["class"] == PERSON_CLASS]["confidence"]

plt.figure()
plt.hist(person_conf, bins=30)
plt.xlabel("Confidence")
plt.ylabel("Number of detections")
plt.title("Person detection confidence distribution")
plt.savefig(OUTPUT_DIR / "person_confidence_hist.png")
plt.close()

# ==============================
# D. Prediction time vs detections
# ==============================
plt.figure()
plt.scatter(
    image_level["num_detections"],
    image_level["prediction_time"],
    alpha=0.6
)
plt.xlabel("Number of detections")
plt.ylabel("Prediction time (s)")
plt.title("Prediction time vs scene complexity")
plt.savefig(OUTPUT_DIR / "time_vs_detections.png")
plt.close()

# ==============================
# E. Non-person class frequency
# ==============================
non_person = df[df["class"] != PERSON_CLASS]
top_classes = non_person["class"].value_counts().head(10)

plt.figure()
top_classes.plot(kind="bar")
plt.ylabel("Number of detections")
plt.title("Top non-person detected classes")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_non_person_classes.png")
plt.close()

# ==============================
# F. Per-celebrity miss rate
# ==============================
celebrity_stats = (
    image_level.groupby("celebrity")["person_detected"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 4))
celebrity_stats.plot(kind="bar")
plt.ylabel("Proportion of images with person detected")
plt.title("Person detection reliability per celebrity")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "celebrity_reliability.png")
plt.close()

print("Plots generated in analysis/plots/")

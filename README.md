# AI-Models_2025

### Requirements

To install all the packages needed, execute:

```bash
pip install -r requirements.txt
```

### Running face_extractor.py

The face extraction script processes images and extracts aligned faces using RetinaFace and dlib landmarks.

**Basic usage:**

```bash
python3 face_extractor.py
```

**With custom arguments:**

```bash
python3 face_extractor.py --input_dir ./working --output_dir ../working_retinaface --output_size 160
```

**Available arguments:**

- `--input_dir` (default: `./working`) - Input directory containing celebrity subdirectories
- `--output_dir` (default: `../working_retinaface`) - Output directory for aligned faces
- `--output_size` (default: `160`) - Output image size in pixels
- `--max_rotation` (default: `10.0`) - Maximum allowed rotation angle in degrees
- `--min_confidence` (default: `0.95`) - Minimum detection confidence score
- `--min_eye_distance` (default: `30.0`) - Minimum eye distance in pixels
- `--max_eye_distance` (default: `200.0`) - Maximum eye distance in pixels
- `--landmark_model` (default: `shape_predictor_68_face_landmarks.dat`) - Path to dlib landmark model
- `--results_csv` (default: `pre/results/results_retina-face.csv`) - Output CSV file for results
- `--needle` (default: `person`) - String to search for in filename (filter)

Graphical User Interface (GUI)

This repository provides an interactive graphical interface to use the trained multi-head face recognition model.
The GUI allows users to upload an image and obtain predictions for:

Celebrity identity (among 105 known celebrities)

Sex classification (Male / Female)

Closest matching celebrity for unknown individuals

The interface is built using Gradio, enabling a clean, browser-based experience similar to modern AI tools.
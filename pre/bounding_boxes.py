from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
import pandas as pd
import time

dataset_directory = "../105_classes_pins_dataset"

# Load a model
model = YOLO("yolo26n.pt")  # load an official model

# Saving the results for evaluation
results = pd.DataFrame({"celebrity": [], # Name of the celebrity on the image
                        "image": [], # Name of the image
                        "class": [], # Class predicted of the detection box
                        "confidence": [], # Confidence score of the detection box
                        "prediction_time": []}) # Prediction time of the image

# Looping through every celeb subdirectory
for celeb in os.listdir(dataset_directory):
    celeb_path = os.path.join(dataset_directory, celeb)
    celeb = celeb.replace(" ", "-")
    os.makedirs(f"../working/{celeb}",  exist_ok=True)

    # Looping through every image in the subdirectory
    for image_file in os.listdir(celeb_path):
        image_path = os.path.join(celeb_path, image_file)
        image_name = image_file.replace(".jpg", "")

        to_crop = Image.open(image_path)

        # Measure prediction time
        start = time.time()

        # Predict with the model
        results = model(image_path)
        
        # Measure prediction time
        end = time.time()
        prediction_time = end - start

        # Access the results
        for result in results:
            # Access each bounding box
            for box in result.boxes:
                # Save the cropped image
                x1 = box.xyxy[0].numpy()[0]
                y1 = box.xyxy[0].numpy()[1]
                x2 = box.xyxy[0].numpy()[2]
                y2 = box.xyxy[0].numpy()[3]
                name = result.names[int(box.cls)]
                cropped = to_crop.crop((x1, y1, x2, y2)).convert('RGB')
                cropped.save(f"working/{celeb}/{image_name}-{name}-nn-bb-{x1}-{y1}-{x2}-{y2}.jpg")

                # Saving the results for evaluation
                r = pd.DataFrame({"celebrity": [celeb],
                     "image": [image_name],
                     "class": [name],
                     "confidence": [box.conf],
                     "prediction_time": [prediction_time]})
                results = pd.concat([results, r])

# Saving results to csv for evaluation
results.to_csv("pre/results/results_bounding_boxes.csv", index=False)

                
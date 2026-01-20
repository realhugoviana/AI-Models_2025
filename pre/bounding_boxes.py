from ultralytics import YOLO # type: ignore
import os
from PIL import Image # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import time

dataset_directory = "../105_classes_pins_dataset"

# Load a model
model = YOLO("yolo26n.pt")  # load an official model

# Save results for evaluation
results_df = pd.DataFrame({"celebrity": [], # Name of the celebrity on the image
                        "image": [], # Name of the image
                        "class": [], # Class predicted of the detection box
                        "confidence": [], # Confidence score of the detection box
                        "prediction_time": []}) # Prediction time of the image

# Access celebs subdirectory
for celeb in os.listdir(dataset_directory):
    celeb_path = os.path.join(dataset_directory, celeb)
    celeb = celeb.replace(" ", "-")
    print(celeb)
    os.makedirs(f"../working/{celeb}",  exist_ok=True)

    # Access images in the subdirectory
    for image_file in os.listdir(celeb_path):
        image_path = os.path.join(celeb_path, image_file)
        image_name = image_file.replace(".jpg", "")
        image_name = image_name.replace(" ", "-")
        print(image_name)

        to_crop = Image.open(image_path)

        # Measure prediction time
        start = time.time()

        # Predict with the model
        results = model(image_path)
        
        # Measure prediction time
        end = time.time()
        prediction_time = end - start
        print(prediction_time)

        # Access results
        for result in results:
            # Access bounding boxes
            for box in result.boxes:
                # Bounding box coordinates
                x1 = box.xyxy[0].numpy()[0]
                y1 = box.xyxy[0].numpy()[1]
                x2 = box.xyxy[0].numpy()[2]
                y2 = box.xyxy[0].numpy()[3]
                # Predicted class
                name = result.names[int(box.cls)]
                # Crop the image to the bounding box coordinates
                cropped = to_crop.crop((x1, y1, x2, y2)).convert('RGB')
                # Save the cropped image
                cropped.save(f"../working/{celeb}/{image_name}-{name}-nn-bb-{x1}-{y1}-{x2}-{y2}.jpg")

                # Save the results for evaluation
                r = pd.DataFrame({"celebrity": [celeb],
                     "image": [image_name],
                     "class": [name],
                     "confidence": [box.conf],
                     "prediction_time": [prediction_time]})
                results_df = pd.concat([results_df, r])

# Save results to csv for evaluation
results_df.to_csv("pre/results/results_bounding_boxes.csv", index=False)

                
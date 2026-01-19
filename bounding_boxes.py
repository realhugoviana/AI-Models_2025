from ultralytics import YOLO
import os
from PIL import Image

dataset_directory = "../Dataset"

# Load a model
model = YOLO("yolo26n.pt")  # load an official model


for celeb in os.listdir(dataset_directory):
    celeb_path = os.path.join(dataset_directory, celeb)
    celeb = celeb.replace(" ", "-")
    os.makedirs(f"working/{celeb}",  exist_ok=True)
    for image_file in os.listdir(celeb_path):
        image_path = os.path.join(celeb_path, image_file)
        image_name = image_file.replace(".jpg", "")

        to_crop = Image.open(image_path)

        # Predict with the model
        results = model(image_path)  # predict on an image

        # Access the results
        for result in results:
            for box in result.boxes:
                x1 = box.xyxy[0].numpy()[0]
                y1 = box.xyxy[0].numpy()[1]
                x2 = box.xyxy[0].numpy()[2]
                y2 = box.xyxy[0].numpy()[3]
                name = result.names[int(box.cls)]
                cropped = to_crop.crop((x1, y1, x2, y2)).convert('RGB')
                cropped.save(f"working/{celeb}/{image_name}-{name}-nn-bb-{x1}-{y1}-{x2}-{y2}.jpg")

# test_result = model("../stevenson.png")

# to_crop = Image.open("../stevenson.png")
# for result in test_result:
#     for box in result.boxes:
#         x1 = box.xyxy[0].numpy()[0]
#         y1 = box.xyxy[0].numpy()[1]
#         x2 = box.xyxy[0].numpy()[2]
#         y2 = box.xyxy[0].numpy()[3]
#         name = result.names[int(box.cls)]
#         # result.save_crop(save_dir="working", file_name=f"salut-{name}-nn-bb-{x1}-{y1}-{x2}-{y2}.jpg")
#         cropped = to_crop.crop((x1, y1, x2, y2))
#         cropped = cropped.convert("RGB")
#         cropped.save(f"working/stevenson-{name}-nn-bb-{x1}-{y1}-{x2}-{y2}.jpg")

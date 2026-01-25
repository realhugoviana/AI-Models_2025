import torch
from ultralytics import YOLO
from src.multi_head_classifier.multi_head_model import MultiHeadFaceRecognitionModel
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classifier = MultiHeadFaceRecognitionModel.load_from_checkpoint(
    checkpoint_path="models/multi_head_classifier.ckpt",
    backbone_weights_path="models/vgg_m_face_bn_dag.pth",
    num_celebrities=105,
    map_location=DEVICE
).eval()


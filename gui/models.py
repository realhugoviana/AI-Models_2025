import torch
from ultralytics import YOLO
from src.multi_head_classifier.multi_head_model import MultiHeadFaceRecognitionModel
from src.one_head_classifier.model import vgg_m_face_bn_dag, FaceEmbeddingModel
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo = YOLO("models/yolo26n.pt").to(DEVICE)

classifier = MultiHeadFaceRecognitionModel.load_from_checkpoint(
    checkpoint_path="models/multi_head_classifier.ckpt",
    backbone_weights_path="models/vgg_m_face_bn_dag.pth",
    num_celebrities=105,
    map_location=DEVICE
).eval()

backbone = vgg_m_face_bn_dag(weights_path="models/vgg_m_face_bn_dag.pth")
embedding_model = FaceEmbeddingModel(backbone)
embedding_model = embedding_model.to(DEVICE).eval()
import torch
from torchvision import transforms
from gui.models import embedding_model, DEVICE
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding(face_np):
    face = Image.fromarray(face_np)
    x = transform(face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = embedding_model(x)

    return emb.squeeze(0)

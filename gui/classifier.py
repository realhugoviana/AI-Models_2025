import torch
from gui.models import classifier

def classify(embedding):
    with torch.no_grad():
        name_logits, sex_logits = classifier.classifier(
            embedding.unsqueeze(0)
        )

    return {
        "name_conf": torch.softmax(name_logits, 1).max().item(),
        "sex": "M" if sex_logits.argmax().item() == 1 else "F"
    }

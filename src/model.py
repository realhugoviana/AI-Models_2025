import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import lightning as L # type: ignore
import torchmetrics # type: ignore

class Vgg_m_face_bn_dag(nn.Module):

    def __init__(self):
        super(Vgg_m_face_bn_dag, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=[7, 7], stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=[5, 5], stride=(2, 2), padding=(1, 1))
        self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=(0, 0), dilation=1, ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn51 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn52 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.bn53 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=[6, 6], stride=(1, 1))
        self.bn54 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=[1, 1], stride=(1, 1))
        self.bn55 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.bn49(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn50(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn51(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn52(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn53(x15)
        x17 = self.relu5(x16)
        x18 = self.pool5(x17)
        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20)
        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24_preflatten = self.relu7(x23)
        x24 = x24_preflatten.view(x24_preflatten.size(0), -1)
        x25 = self.fc8(x24)
        return x25

def vgg_m_face_bn_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_m_face_bn_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

class FaceEmbeddingModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc8 = nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        embeddings = F.normalize(features, p=2, dim=1)
        return embeddings
    
class Classifier(nn.Module):
    def __init__(self, input_dim=4096, output_dim=105, learning_rate=1e-3, s=64, weight_decay=1e-4):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.s = s

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.fc.weight, dim=1)
        return self.s * F.linear(x, w)
    
class FaceRecognitionModel(L.LightningModule):
    def __init__(self, backbone_weights_path, num_classes, learning_rate=1e-3, s=64, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        backbone = vgg_m_face_bn_dag(weights_path=backbone_weights_path)
        self.embedding_model = FaceEmbeddingModel(backbone)

        # Freeze backbone explicitly
        for p in self.embedding_model.parameters():
            p.requires_grad = False

        self.classifier = Classifier(
            input_dim=4096,
            output_dim=num_classes,
            learning_rate=learning_rate,
            s=s,
            weight_decay=weight_decay
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes
        )

    def forward(self, x):
        with torch.no_grad():
            embeddings = self.embedding_model(x)
        logits = self.classifier(embeddings)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.train_acc(logits.argmax(dim=1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.val_acc(logits.argmax(dim=1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.test_acc(logits.argmax(dim=1), y)
        self.log("test_loss", loss)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.classifier.parameters(),  # HEAD ONLY
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
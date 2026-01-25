import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics

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

class MultiHeadClassifier(nn.Module):
    """
    Two-head classifier for celebrity name and sex prediction
    """
    def __init__(self, input_dim=4096, num_celebrities=105, num_sex_classes=2, s_name=64, s_sex=64):
        super().__init__()
        # Celebrity name classifier
        self.fc_name = nn.Linear(input_dim, num_celebrities, bias=False)
        self.s_name = s_name
        
        # Sex classifier
        self.fc_sex = nn.Linear(input_dim, num_sex_classes, bias=False)
        self.s_sex = s_sex

    def forward(self, x):
        # Normalize input
        x = F.normalize(x, dim=1)
        
        # Celebrity name prediction
        w_name = F.normalize(self.fc_name.weight, dim=1)
        logits_name = self.s_name * F.linear(x, w_name)
        
        # Sex prediction
        w_sex = F.normalize(self.fc_sex.weight, dim=1)
        logits_sex = self.s_sex * F.linear(x, w_sex)
        
        return logits_name, logits_sex

class MultiHeadFaceRecognitionModel(L.LightningModule):
    def __init__(
        self, 
        backbone_weights_path, 
        num_celebrities, 
        num_sex_classes=2,
        learning_rate=1e-3, 
        s_name=64,
        s_sex=64,
        weight_decay=1e-4,
        loss_weight_name=1.0,
        loss_weight_sex=1.0
    ):
        super().__init__()
        self.save_hyperparameters()

        backbone = vgg_m_face_bn_dag(weights_path=backbone_weights_path)
        self.embedding_model = FaceEmbeddingModel(backbone)

        # Freeze backbone explicitly
        for p in self.embedding_model.parameters():
            p.requires_grad = False

        self.classifier = MultiHeadClassifier(
            input_dim=4096,
            num_celebrities=num_celebrities,
            num_sex_classes=num_sex_classes,
            s_name=s_name,
            s_sex=s_sex
        )

        self.criterion_name = nn.CrossEntropyLoss()
        self.criterion_sex = nn.CrossEntropyLoss()

        # Metrics for celebrity name (weighted macro averaging)
        self.train_acc_name = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_celebrities
        )
        self.val_acc_name = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_celebrities
        )
        self.test_acc_name = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_celebrities
        )
        
        # Celebrity name: Precision, Recall, F1
        self.test_precision_name = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_celebrities,
            average='macro'
        )
        self.test_recall_name = torchmetrics.Recall(
            task="multiclass",
            num_classes=num_celebrities,
            average='macro'
        )
        self.test_f1_name = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_celebrities,
            average='macro'
        )

        # Metrics for sex
        self.train_acc_sex = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_sex_classes
        )
        self.val_acc_sex = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_sex_classes
        )
        self.test_acc_sex = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_sex_classes
        )
        
        # Sex: Precision, Recall, F1
        self.test_precision_sex = torchmetrics.Precision(
            task="multiclass",
            num_classes=num_sex_classes,
            average='macro'
        )
        self.test_recall_sex = torchmetrics.Recall(
            task="multiclass",
            num_classes=num_sex_classes,
            average='macro'
        )
        self.test_f1_sex = torchmetrics.F1Score(
            task="multiclass",
            num_classes=num_sex_classes,
            average='macro'
        )
        
        # Confusion matrices for detailed analysis
        self.test_confmat_name = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=num_celebrities
        )
        self.test_confmat_sex = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=num_sex_classes
        )
        
        # Store predictions for cross-performance analysis
        self.test_predictions_name = []
        self.test_predictions_sex = []
        self.test_targets_name = []
        self.test_targets_sex = []

    def forward(self, x):
        with torch.no_grad():
            embeddings = self.embedding_model(x)
        logits_name, logits_sex = self.classifier(embeddings)
        return logits_name, logits_sex

    def training_step(self, batch, batch_idx):
        x, y_name, y_sex = batch
        logits_name, logits_sex = self(x)
        
        loss_name = self.criterion_name(logits_name, y_name)
        loss_sex = self.criterion_sex(logits_sex, y_sex)
        
        # Weighted combined loss
        loss = (self.hparams.loss_weight_name * loss_name + 
                self.hparams.loss_weight_sex * loss_sex)

        # Accuracy metrics
        acc_name = self.train_acc_name(logits_name.argmax(dim=1), y_name)
        acc_sex = self.train_acc_sex(logits_sex.argmax(dim=1), y_sex)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_loss_name", loss_name, sync_dist=True)
        self.log("train_loss_sex", loss_sex, sync_dist=True)
        self.log("train_acc_name", acc_name, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc_sex", acc_sex, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_name, y_sex = batch
        logits_name, logits_sex = self(x)
        
        loss_name = self.criterion_name(logits_name, y_name)
        loss_sex = self.criterion_sex(logits_sex, y_sex)
        
        loss = (self.hparams.loss_weight_name * loss_name + 
                self.hparams.loss_weight_sex * loss_sex)

        acc_name = self.val_acc_name(logits_name.argmax(dim=1), y_name)
        acc_sex = self.val_acc_sex(logits_sex.argmax(dim=1), y_sex)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_loss_name", loss_name, sync_dist=True)
        self.log("val_loss_sex", loss_sex, sync_dist=True)
        self.log("val_acc_name", acc_name, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc_sex", acc_sex, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y_name, y_sex = batch
        logits_name, logits_sex = self(x)
        
        loss_name = self.criterion_name(logits_name, y_name)
        loss_sex = self.criterion_sex(logits_sex, y_sex)
        
        loss = (self.hparams.loss_weight_name * loss_name + 
                self.hparams.loss_weight_sex * loss_sex)

        # Get predictions
        preds_name = logits_name.argmax(dim=1)
        preds_sex = logits_sex.argmax(dim=1)

        # Update metrics
        acc_name = self.test_acc_name(preds_name, y_name)
        acc_sex = self.test_acc_sex(preds_sex, y_sex)
        
        # Update precision, recall, F1
        self.test_precision_name(preds_name, y_name)
        self.test_recall_name(preds_name, y_name)
        self.test_f1_name(preds_name, y_name)
        
        self.test_precision_sex(preds_sex, y_sex)
        self.test_recall_sex(preds_sex, y_sex)
        self.test_f1_sex(preds_sex, y_sex)
        
        # Update confusion matrices
        self.test_confmat_name(preds_name, y_name)
        self.test_confmat_sex(preds_sex, y_sex)

        # Store predictions for cross-performance analysis
        self.test_predictions_name.extend(preds_name.cpu().tolist())
        self.test_predictions_sex.extend(preds_sex.cpu().tolist())
        self.test_targets_name.extend(y_name.cpu().tolist())
        self.test_targets_sex.extend(y_sex.cpu().tolist())

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_loss_name", loss_name, sync_dist=True)
        self.log("test_loss_sex", loss_sex, sync_dist=True)
        self.log("test_acc_name", acc_name, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc_sex", acc_sex, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def on_test_epoch_end(self):
        """Compute and log final metrics at end of test epoch"""
        import numpy as np
        
        # Compute final metrics
        precision_name = self.test_precision_name.compute()
        recall_name = self.test_recall_name.compute()
        f1_name = self.test_f1_name.compute()
        
        precision_sex = self.test_precision_sex.compute()
        recall_sex = self.test_recall_sex.compute()
        f1_sex = self.test_f1_sex.compute()
        
        # Log metrics
        self.log("test_precision_name", precision_name)
        self.log("test_recall_name", recall_name)
        self.log("test_f1_name", f1_name)
        
        self.log("test_precision_sex", precision_sex)
        self.log("test_recall_sex", recall_sex)
        self.log("test_f1_sex", f1_sex)
        
        # Print detailed results
        print("\n" + "="*70)
        print("CELEBRITY NAME CLASSIFIER METRICS (Macro Average)")
        print("="*70)
        print(f"Precision: {precision_name:.4f}")
        print(f"Recall:    {recall_name:.4f}")
        print(f"F1-Score:  {f1_name:.4f}")
        
        print("\n" + "="*70)
        print("SEX CLASSIFIER METRICS (Macro Average)")
        print("="*70)
        print(f"Precision: {precision_sex:.4f}")
        print(f"Recall:    {recall_sex:.4f}")
        print(f"F1-Score:  {f1_sex:.4f}")
        
        # Cross-performance analysis
        print("\n" + "="*70)
        print("CROSS-PERFORMANCE ANALYSIS")
        print("="*70)
        
        preds_name = np.array(self.test_predictions_name)
        preds_sex = np.array(self.test_predictions_sex)
        targets_name = np.array(self.test_targets_name)
        targets_sex = np.array(self.test_targets_sex)
        
        # 1. When name is correct, sex accuracy
        name_correct = (preds_name == targets_name)
        if name_correct.sum() > 0:
            sex_acc_when_name_correct = (preds_sex[name_correct] == targets_sex[name_correct]).mean()
            print(f"\nSex accuracy when celebrity name is CORRECT: {sex_acc_when_name_correct:.4f}")
        
        # 2. When name is wrong, sex accuracy
        name_wrong = ~name_correct
        if name_wrong.sum() > 0:
            sex_acc_when_name_wrong = (preds_sex[name_wrong] == targets_sex[name_wrong]).mean()
            print(f"Sex accuracy when celebrity name is WRONG:   {sex_acc_when_name_wrong:.4f}")
        
        # 3. When sex is correct, name accuracy
        sex_correct = (preds_sex == targets_sex)
        if sex_correct.sum() > 0:
            name_acc_when_sex_correct = (preds_name[sex_correct] == targets_name[sex_correct]).mean()
            print(f"\nCelebrity accuracy when sex is CORRECT: {name_acc_when_sex_correct:.4f}")
        
        # 4. When sex is wrong, name accuracy
        sex_wrong = ~sex_correct
        if sex_wrong.sum() > 0:
            name_acc_when_sex_wrong = (preds_name[sex_wrong] == targets_name[sex_wrong]).mean()
            print(f"Celebrity accuracy when sex is WRONG:   {name_acc_when_sex_wrong:.4f}")
        
        # 5. Both correct
        both_correct = name_correct & sex_correct
        both_correct_pct = both_correct.mean()
        print(f"\nBoth predictions correct: {both_correct_pct:.4f} ({both_correct.sum()}/{len(targets_name)})")
        
        # 6. Only one correct
        only_name_correct = name_correct & ~sex_correct
        only_sex_correct = ~name_correct & sex_correct
        
        print(f"Only celebrity correct:   {only_name_correct.mean():.4f} ({only_name_correct.sum()}/{len(targets_name)})")
        print(f"Only sex correct:         {only_sex_correct.mean():.4f} ({only_sex_correct.sum()}/{len(targets_name)})")
        
        # 7. Both wrong
        both_wrong = ~name_correct & ~sex_correct
        print(f"Both predictions wrong:   {both_wrong.mean():.4f} ({both_wrong.sum()}/{len(targets_name)})")
        
        # Per-sex analysis
        print("\n" + "="*70)
        print("PER-SEX CELEBRITY RECOGNITION PERFORMANCE")
        print("="*70)
        for sex_idx in range(self.hparams.num_sex_classes):
            sex_mask = (targets_sex == sex_idx)
            if sex_mask.sum() > 0:
                name_acc_for_sex = (preds_name[sex_mask] == targets_name[sex_mask]).mean()
                sex_label = "Male" if sex_idx == 0 else "Female"  # Adjust based on your mapping
                print(f"Celebrity accuracy for {sex_label}: {name_acc_for_sex:.4f} ({sex_mask.sum()} samples)")
        
        print("="*70 + "\n")
        
        # Reset for next epoch
        self.test_predictions_name = []
        self.test_predictions_sex = []
        self.test_targets_name = []
        self.test_targets_sex = []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
import optuna  # type: ignore
import torch   # type: ignore
import lightning as L  # type: ignore

from optuna.integration import PyTorchLightningPruningCallback  # type: ignore
from lightning.pytorch.loggers import TensorBoardLogger  # type: ignore
from lightning.pytorch.callbacks import ModelCheckpoint  # type: ignore

from dataset import VGGFaceDataModule
from model import FaceRecognitionModel

storage = optuna.storages.RDBStorage(
    url="sqlite:///optuna_face_recognition.db",
    engine_kwargs={"connect_args": {"timeout": 60}},
)


def objective(trial):
    lr = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)
    s = trial.suggest_categorical("s", [16, 32, 48, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)

    model = FaceRecognitionModel(
        backbone_weights_path="vgg_m_face_bn_dag.pth",
        num_classes=105,
        learning_rate=lr,
        s=s,
        weight_decay=weight_decay,
    )

    logger = TensorBoardLogger(
        save_dir="logs",
        name="optuna_trials",
        version=f"trial_{trial.number}",
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="epoch{epoch:02d}-val_acc{val_acc:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        precision=32,
        logger=logger,
        callbacks=[
            checkpoint_cb,
            PyTorchLightningPruningCallback(trial, monitor="val_acc"),
        ],
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    val_acc = trainer.callback_metrics.get("val_acc")
    return val_acc.item() if val_acc is not None else 0.0


if __name__ == "__main__":
    L.seed_everything(42, workers=True)

    datamodule = VGGFaceDataModule(
        data_dir="../working_retinaface",
        batch_size=32, 
        num_workers=8
    )

    study = optuna.create_study(
        study_name="face_recognition_ddp",
        storage=storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=30)

    print("\nBest validation accuracy:", study.best_value)
    print("Best hyperparameters:")
    print(study.best_trial.params)

    best = study.best_trial.params

    final_model = FaceRecognitionModel(
        backbone_weights_path="vgg_m_face_bn_dag.pth",
        num_classes=105,
        learning_rate=best["learning_rate"],
        s=best["s"],
        weight_decay=best["weight_decay"],
    )

    final_logger = TensorBoardLogger(
        save_dir="logs",
        name="final_model",
    )

    final_checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-epoch{epoch:02d}-val_acc{val_acc:.4f}",
    )

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        precision=32,
        logger=final_logger,
        callbacks=[final_checkpoint],
        enable_checkpointing=True,
    )

    trainer.fit(final_model, datamodule=datamodule)
    trainer.test(final_model, datamodule=datamodule)

    print("\nBest model checkpoint:")
    print(final_checkpoint.best_model_path)

    best_model = FaceRecognitionModel.load_from_checkpoint(
        final_checkpoint.best_model_path
    )

    torch.save(
        best_model.classifier.state_dict(),
        "classifier_head.pth",
    )

    print("Classifier head saved to classifier_head.pth")
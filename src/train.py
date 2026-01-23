import optuna # type: ignore
import lightning as L # type: ignore
from optuna.integration import PyTorchLightningPruningCallback # type: ignore

from dataset import VGGFaceDataModule
from model import Classifier, FaceRecognitionModel

def objective(trial):
    lr = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)
    s = trial.suggest_categorical("s", [16, 32, 48, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)

    model = FaceRecognitionModel(
        backbone_weights_path="vgg_m_face_bn_dag.pth",
        num_classes=105,
        learning_rate=lr,
        s=s,
        weight_decay=weight_decay
    )

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        callbacks=[
            PyTorchLightningPruningCallback(
                trial,
                monitor="val_acc"
            )
        ],
    )

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_acc"].item()

if __name__ == "__main__":
    L.seed_everything(42, workers=True)

    datamodule = VGGFaceDataModule(
        data_dir="../working_retinaface",
        batch_size=32,
        num_workers=4
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3
        )
    )

    study.optimize(objective, n_trials=30)

    print("Best validation accuracy:", study.best_value)
    print("Best hyperparameters:")
    print(study.best_trial.params)

    best = study.best_trial.params

    final_model = FaceRecognitionModel(
        backbone_weights_path="vgg_m_face_bn_dag.pth",
        num_classes=105,
        learning_rate=best["learning_rate"],
        s=best["s"],
        weight_decay=best["weight_decay"]
    )

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="auto",
        devices=1
    )

    trainer.fit(final_model, datamodule=datamodule)
    trainer.test(final_model, datamodule=datamodule)
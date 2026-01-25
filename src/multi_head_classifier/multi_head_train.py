import optuna
import torch
import lightning as L

from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from multi_head_dataset import MultiHeadVGGFaceDataModule
from multi_head_model import MultiHeadFaceRecognitionModel

storage = optuna.storages.RDBStorage(
    url="sqlite:///optuna_multihead_face_recognition.db",
    engine_kwargs={"connect_args": {"timeout": 60}},
)


def objective(trial):
    lr = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)
    s_name = trial.suggest_categorical("s_name", [16, 32, 48, 64])
    s_sex = trial.suggest_categorical("s_sex", [16, 32, 48, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
    loss_weight_name = trial.suggest_float("loss_weight_name", 0.5, 2.0)
    loss_weight_sex = trial.suggest_float("loss_weight_sex", 0.5, 2.0)

    model = MultiHeadFaceRecognitionModel(
        backbone_weights_path="vgg_m_face_bn_dag.pth",
        num_celebrities=datamodule.name_to_idx.__len__(),
        num_sex_classes=datamodule.sex_to_idx.__len__(),
        learning_rate=lr,
        s_name=s_name,
        s_sex=s_sex,
        weight_decay=weight_decay,
        loss_weight_name=loss_weight_name,
        loss_weight_sex=loss_weight_sex,
    )

    logger = TensorBoardLogger(
        save_dir="logs",
        name="optuna_trials_multihead",
        version=f"trial_{trial.number}",
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc_name",  # Monitor celebrity name accuracy
        mode="max",
        save_top_k=1,
        filename="epoch{epoch:02d}-val_acc_name{val_acc_name:.4f}",
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
            PyTorchLightningPruningCallback(trial, monitor="val_acc_name"),
        ],
        enable_checkpointing=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    # Return combined metric (weighted average of both accuracies)
    val_acc_name = trainer.callback_metrics.get("val_acc_name")
    val_acc_sex = trainer.callback_metrics.get("val_acc_sex")
    
    if val_acc_name is not None and val_acc_sex is not None:
        # Weighted combination: name is more important
        combined_metric = 0.7 * val_acc_name.item() + 0.3 * val_acc_sex.item()
        return combined_metric
    else:
        return 0.0


if __name__ == "__main__":
    L.seed_everything(42, workers=True)

    # Initialize datamodule
    datamodule = MultiHeadVGGFaceDataModule(
        data_dir="./working_retinaface",
        batch_size=32,
        num_workers=8,
        n_train_per_class=12,  # 12 training samples per celebrity
        n_val_per_class=2,     # 2 validation samples per celebrity
        # Test samples = remaining (e.g., Jeff Bezos: 17 - 12 - 2 = 3 test)
    )
    
    # Setup to get number of classes
    datamodule.setup()
    
    print(f"\nNumber of celebrities: {len(datamodule.name_to_idx)}")
    print(f"Number of sex classes: {len(datamodule.sex_to_idx)}")
    print(f"Celebrity mapping sample: {list(datamodule.name_to_idx.items())[:5]}")
    print(f"Sex mapping: {datamodule.sex_to_idx}\n")

    # Create Optuna study
    study = optuna.create_study(
        study_name="multihead_face_recognition_ddp",
        storage=storage,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
        load_if_exists=True,
    )
    
    max_trials = 30
    existing_trials = len(study.trials)
    remaining_trials = max(0, max_trials - existing_trials)

    if remaining_trials > 0:
        print(f"Running {remaining_trials} trials...\n")
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("Study already completed.")

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best combined metric: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Train final model with best hyperparameters
    best = study.best_trial.params

    final_model = MultiHeadFaceRecognitionModel(
        backbone_weights_path="vgg_m_face_bn_dag.pth",
        num_celebrities=len(datamodule.name_to_idx),
        num_sex_classes=len(datamodule.sex_to_idx),
        learning_rate=best["learning_rate"],
        s_name=best["s_name"],
        s_sex=best["s_sex"],
        weight_decay=best["weight_decay"],
        loss_weight_name=best["loss_weight_name"],
        loss_weight_sex=best["loss_weight_sex"],
    )

    final_logger = TensorBoardLogger(
        save_dir="logs",
        name="final_multihead_model",
    )

    final_checkpoint = ModelCheckpoint(
        monitor="val_acc_name",
        mode="max",
        save_top_k=1,
        filename="best-epoch{epoch:02d}-val_acc_name{val_acc_name:.4f}-val_acc_sex{val_acc_sex:.4f}",
    )

    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)

    trainer = L.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision=32,
        logger=final_logger,
        callbacks=[final_checkpoint],
        enable_checkpointing=True,
    )

    trainer.fit(final_model, datamodule=datamodule)
    trainer.test(final_model, datamodule=datamodule)

    print("\n" + "="*60)
    print("FINAL MODEL SAVED")
    print("="*60)
    print(f"Best checkpoint: {final_checkpoint.best_model_path}")

    # Load and save the classifier heads
    best_model = MultiHeadFaceRecognitionModel.load_from_checkpoint(
        final_checkpoint.best_model_path
    )

    # Save both classifier heads
    torch.save(
        best_model.classifier.state_dict(),
        "multihead_classifier.pth",
    )

    # Save individual heads for easier loading
    torch.save(
        best_model.classifier.fc_name.state_dict(),
        "classifier_head_name.pth",
    )
    
    torch.save(
        best_model.classifier.fc_sex.state_dict(),
        "classifier_head_sex.pth",
    )

    # Save the mappings for inference
    torch.save({
        'name_to_idx': datamodule.name_to_idx,
        'idx_to_name': datamodule.idx_to_name,
        'sex_to_idx': datamodule.sex_to_idx,
        'idx_to_sex': datamodule.idx_to_sex,
    }, "class_mappings.pth")

    print("\nSaved files:")
    print("  - multihead_classifier.pth (complete classifier)")
    print("  - classifier_head_name.pth (celebrity name head)")
    print("  - classifier_head_sex.pth (sex head)")
    print("  - class_mappings.pth (class index mappings)")
    print("\n" + "="*60)

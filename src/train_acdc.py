from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
import torch
from torch.utils.data import DataLoader


from segment2d import *
from lightning.pytorch.loggers import WandbLogger
import os
import csv
import wandb
import argparse
import glob


torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--accelerator", type=str, default="gpu")
parser.add_argument("--devices", type=int, nargs="+", default=[0], help="list of devices")
parser.add_argument("--epoch", type=int, default=3000)
parser.add_argument("-bs", "--batch_size", type=int, default=8)
parser.add_argument("-lr,", "--learning_rate", type=float, default=0.001)
parser.add_argument(
    "-flr", "--factor_lr", type=float, default=0.5, help="new learning rate = factor_lr * old learning rate"
)
parser.add_argument("-plr", "--patience_lr", type=int, default=20, help="factor of learning rate when it is reduced")
parser.add_argument("-pes", "--patience_es", type=int, default=200, help="early stopping")

parser.add_argument("--mix_precision", type=str, default="16-mixed", help="16-mixed or 32")
args = parser.parse_args()
# Main function
if __name__ == "__main__":

    os.makedirs(f"./weights_ACDC/", exist_ok=True)
    save_dir = f"./weights_ACDC/"

    num_classes = 4
    model = FCDenseNet(in_channels=cfg.DATA.INDIM_MODEL, n_classes=num_classes)
    class_weight = [0.27, 10.00, 8.91, 8.94]  #


    with open("csv_files/ACDC_val.csv", mode="r") as f:
        reader = csv.DictReader(f)
        list_val_subject = [row["path"] for row in reader]


    segmenter = Segmenter_ACDC(
        model,
        class_weight,
        num_classes,
        args.learning_rate,
        args.factor_lr,
        args.patience_lr,
    )
    list_train_subject = glob.glob(f"ACDC_train/*.npz")
    train_dataset = Image_Loader(list_subject=list_train_subject)

    val_dataset = Test_Volume_Loader(list_val_subject)

    # If wandb_logger is True, create a WandbLogger object
    if cfg.TRAIN.WANDB:
        wandb_logger = WandbLogger(
            project="ACDC",
            name=f"tiramisu",
            resume="allow",
        )
    else:
        wandb_logger = False
    # Define data loaders for the training and test data
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        drop_last=True,
        prefetch_factor=cfg.TRAIN.PREFETCH_FACTOR,
    )

    # Initialize a ModelCheckpoint callback to save the model weights after each epoch
    check_point = ModelCheckpoint(
        save_dir,
        filename="dice_{val_dice:0.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=cfg.TRAIN.SAVE_TOP_K,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
        save_last=True,
    )

    # Initialize a LearningRateMonitor callback to log the learning rate during training
    lr_monitor = LearningRateMonitor(logging_interval=None)
    # Initialize a EarlyStopping callback to stop training if the validation loss does not improve for a certain number of epochs
    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=args.patience_es,
        verbose=True,
        strict=False,
    )

    print("class_weight:", class_weight)
    print("Use loss:", cfg.TRAIN.LOSS)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)
    print("Factor LR:", args.factor_lr)
    print("Patience LR:", args.patience_lr)
    print("Patience ES:", args.patience_es)
    print("Epoch:", args.epoch)
    print("Mix precision:", args.mix_precision)
    print("Accelerator:", args.accelerator)
    print("Devices:", args.devices)

    # Define a dictionary with the parameters for the Trainer object
    PARAMS_TRAINER = {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "benchmark": True,
        "enable_progress_bar": True,
        # "overfit_batches" :5,
        "logger": wandb_logger,
        "callbacks": [check_point, early_stopping, lr_monitor],
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 1,
        "max_epochs": args.epoch,
        "precision": args.mix_precision,
    }

    # Initialize a Trainer object with the specified parameters
    trainer = pl.Trainer(**PARAMS_TRAINER)
    # Get a list of file paths for all non-hidden files in the SAVE_DIR directory
    checkpoint_paths = glob.glob(os.path.join(save_dir, "*.ckpt"))
    checkpoint_paths.sort()
    # If there are checkpoint paths and the load_checkpoint flag is set to True
    if checkpoint_paths and cfg.TRAIN.LOAD_CHECKPOINT:
        # Select the second checkpoint in the list (index 0)
        checkpoint = checkpoint_paths[cfg.TRAIN.IDX_CHECKPOINT]
        print(f"load checkpoint: {checkpoint}")
        # Load the model weights from the selected checkpoint
        segmenter = Segmenter_ACDC.load_from_checkpoint(
            checkpoint_path=checkpoint,
            model=model,
            class_weight=class_weight,
            num_classes=num_classes,
            learning_rate=args.learning_rate,
            factor_lr=args.factor_lr,
            patience_lr=args.patience_lr,
        )

    # Train the model using the train_dataset and test_dataset data loaders
    trainer.fit(segmenter, train_loader, val_dataset)

    wandb.finish()

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
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("-bs", "--batch_size", type=int, default=8)
parser.add_argument("-lr,", "--learning_rate", type=float, default=0.001)
parser.add_argument("-flr", "--factor_lr", type=float, default=0.5, help="new learning rate = factor_lr * old learning rate")
parser.add_argument("-plr", "--patience_lr", type=int, default=50, help="factor of learning rate when it is reduced")
parser.add_argument("-pes", "--patience_es", type=int, default=27, help="early stopping")
parser.add_argument("--save_dir", type=str, default="./weights/")
parser.add_argument("--type_train", type=str, default="train_test")
parser.add_argument("--mix_precision", type=str, default="16-mixed", help="16-mixed or 32")
args = parser.parse_args()
# Main function
if __name__ == "__main__":

    os.makedirs("./weights/", exist_ok=True)

    model = FCDenseNet(in_channels=cfg.DATA.INDIM_MODEL, n_classes=cfg.DATA.NUM_CLASS)

    if args.type_train == "train_test":
        # read csv file
        with open("./test.csv", mode="r") as f:
            reader = csv.DictReader(f)
            list_test_subject = [row["path"] for row in reader]
    elif args.type_train == "train_test_val":
        with open("./val.csv", mode="r") as f:
            reader = csv.DictReader(f)
            list_test_subject = [row["path"] for row in reader]
    else:
        raise ValueError("type_train must be train_test or train_test_val")

    segmenter = Segmenter(
        model,
        cfg.DATA.CLASS_WEIGHT,
        cfg.DATA.NUM_CLASS,
        args.learning_rate,
        args.factor_lr,
        args.patience_lr,
    )
    list_train_subject = glob.glob("./emidec_train/*")
    train_dataset = EMIDEC_Loader(list_subject=list_train_subject)

    test_dataset = EMIDEC_Test_Loader(list_test_subject)

    # If wandb_logger is True, create a WandbLogger object
    if cfg.TRAIN.WANDB:
        wandb_logger = WandbLogger(
            project="emidec",
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
    check_point_myo = ModelCheckpoint(
        args.save_dir,
        filename="myo_{val_dice_MYO:0.4f}",
        monitor="val_dice_MYO",
        mode="max",
        save_top_k=cfg.TRAIN.SAVE_TOP_K,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
        save_last=True,
    )

    check_point_lv = ModelCheckpoint(
        args.save_dir,
        filename="lv_{val_dice_LV:0.4f}",
        monitor="val_dice_LV",
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
        monitor="val_score",
        mode="max",
        patience=args.patience_es,
        verbose=True,
        strict=False,
    )

    print("class_weight:", cfg.DATA.CLASS_WEIGHT)
    print("Use loss:", cfg.TRAIN.LOSS)

    # Define a dictionary with the parameters for the Trainer object
    PARAMS_TRAINER = {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "benchmark": True,
        "enable_progress_bar": True,
        # "overfit_batches" :5,
        "logger": wandb_logger,
        "callbacks": [check_point_myo, check_point_lv, early_stopping, lr_monitor],
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 1,
        "max_epochs": args.epoch,
        "precision": args.mix_precision,
    }

    # Initialize a Trainer object with the specified parameters
    trainer = pl.Trainer(**PARAMS_TRAINER)
    # Get a list of file paths for all non-hidden files in the SAVE_DIR directory
    checkpoint_paths = glob.glob(os.path.join(args.save_dir, "*.ckpt"))
    checkpoint_paths.sort()
    # If there are checkpoint paths and the load_checkpoint flag is set to True
    if checkpoint_paths and cfg.TRAIN.LOAD_CHECKPOINT:
        # Select the second checkpoint in the list (index 0)
        checkpoint = checkpoint_paths[cfg.TRAIN.IDX_CHECKPOINT]
        print(f"load checkpoint: {checkpoint}")
        # Load the model weights from the selected checkpoint
        segmenter = Segmenter.load_from_checkpoint(
            checkpoint_path=checkpoint,
            model=model,
            class_weight=cfg.DATA.CLASS_WEIGHT,
            num_classes=cfg.DATA.NUM_CLASS,
            learning_rate=args.learning_rate,
            factor_lr=args.factor_lr,
            patience_lr=args.patience_lr,
        )

    # Train the model using the train_dataset and test_dataset data loaders
    trainer.fit(segmenter, train_loader, test_dataset)

    wandb.finish()

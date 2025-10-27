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
parser.add_argument("--epoch", type=int, default=10000000)
parser.add_argument("-bs", "--batch_size", type=int, default=8)
parser.add_argument("-lr,", "--learning_rate", type=float, default=0.001)
parser.add_argument(
    "-flr", "--factor_lr", type=float, default=0.5, help="new learning rate = factor_lr * old learning rate"
)
parser.add_argument("-plr", "--patience_lr", type=int, default=200, help="factor of learning rate when it is reduced")
parser.add_argument("-pes", "--patience_es", type=int, default=6500, help="early stopping")

parser.add_argument("--type_train", type=str, default="train_test")
parser.add_argument("--mix_precision", type=str, default="16-mixed", help="16-mixed or 32")
parser.add_argument("--task", type=str, default="train_full", help="train_full, train_combine_myo")
parser.add_argument("--use_EMIDEC", type=bool, default=False)
args = parser.parse_args()
# Main function
if __name__ == "__main__":
    if args.task == "train_full":
        num_classes = 4
    elif args.task == "train_combine_myo":
        num_classes = 3
    else:
        raise ValueError("task must be train_full or train_combine_myo")
    
    if not args.use_EMIDEC:
        print("use MnM only")
        save_dir = f"./weights_MnM_{args.task}/"
    else:
        print("use MnM+EMIDEC")
        save_dir = f"./weights_MnM_EMIDEC_{args.task}/"
    os.makedirs(save_dir, exist_ok=True)

    model = FCDenseNet(in_channels=cfg.DATA.INDIM_MODEL, n_classes=num_classes)
    cfg.TRAIN.TASK = args.task 
    
    if args.task== "train_full":
        cfg.DATA.CLASS_WEIGHT = [0.1, 12, 12, 12]  #
    else:
        cfg.DATA.CLASS_WEIGHT = [0.1, 12, 12]  #

    if args.type_train == "train_test":
        # read csv file if use_EMIDEC is False, otherwise combine read csv file of EMIDEC and MnM
        with open(f"./csv_files/test_MnM_{args.task}.csv", mode="r") as f:
            reader = csv.DictReader(f)
            list_test_subject = [row["path"] for row in reader]
        if args.use_EMIDEC:
            with open(f"./csv_files/EMIDEC_test_{args.task}.csv", mode="r") as f:
                reader = csv.DictReader(f)
                list_test_subject += [row["path"] for row in reader]    
    elif args.type_train == "train_test_val":
        with open(f"./csv_files/val_MnM_{args.task}.csv", mode="r") as f:
            reader = csv.DictReader(f)
            list_test_subject = [row["path"] for row in reader]
        if args.use_EMIDEC:
            with open(f"./csv_files/EMIDEC_val_{args.task}.csv", mode="r") as f:
                reader = csv.DictReader(f)
                list_test_subject += [row["path"] for row in reader]
    else:
        raise ValueError("type_train must be train_test or train_test_val")

    segmenter = Segmenter_MnM(
        model,
        cfg.DATA.CLASS_WEIGHT,
        num_classes,
        args.learning_rate,
        args.factor_lr,
        args.patience_lr,
    )
    if args.task == "train_full":
        list_train_subject = glob.glob(f"./MnM_train_full/*")
    # if use_EMIDEC is True, then add EMIDEC train data to list_train_subject only work for train_combine_myo
    elif args.task == "train_combine_myo":
        list_train_subject = glob.glob(f"./MnM_train_myo/*")
        if args.use_EMIDEC:
            list_train_subject += glob.glob(f"./emidec_train_combine_myo/*")
    else:
        raise ValueError("task must be train_full or train_combine_myo")

    train_dataset = Image_Loader(list_subject=list_train_subject)

    test_dataset = Test_Volume_Loader(list_test_subject)

    # If wandb_logger is True, create a WandbLogger object
    if cfg.TRAIN.WANDB:
        if args.use_EMIDEC:
            wandb_logger = WandbLogger(
                project="MnM+EMIDEC",
                name=f"tiramisu_{args.task}",
                resume="allow",
            )
        else:
            wandb_logger = WandbLogger(
                project="MnM",
                name=f"tiramisu_{args.task}",
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
        segmenter = Segmenter_MnM.load_from_checkpoint(
            checkpoint_path=checkpoint,
            model=model,
            class_weight=cfg.DATA.CLASS_WEIGHT,
            num_classes=num_classes,
            learning_rate=args.learning_rate,
            factor_lr=args.factor_lr,
            patience_lr=args.patience_lr,
        )

    # Train the model using the train_dataset and test_dataset data loaders
    trainer.fit(segmenter, train_loader, test_dataset)

    wandb.finish()

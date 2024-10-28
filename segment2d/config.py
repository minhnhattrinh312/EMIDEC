# Path: segment2d/config.py
from yacs.config import CfgNode as CN

cfg = CN()
cfg.DATA = CN()
cfg.TRAIN = CN()
cfg.SYS = CN()
cfg.OPT = CN()
cfg.DIRS = CN()
cfg.PREDICT = CN()


cfg.DATA.NUM_CLASS = 5
cfg.DATA.CLASS_WEIGHT = [0.1, 2, 2, 17, 140]  #
cfg.DATA.DIM2PAD = [128, 128]
cfg.DATA.INDIM_MODEL = 1

cfg.TRAIN.TASK = "emidec"  #
# "active_focal" or "focal_contour" or "active_contour"
# TverskyLoss,  CrossEntropy, DiceLoss, MSELoss
cfg.TRAIN.LOSS = "focal_contour"

cfg.TRAIN.DISTINCT_SUBJECT = True
cfg.TRAIN.FREEZE = True
cfg.TRAIN.EVA_N_EPOCHS = 2
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.EPOCHS = 1000
cfg.TRAIN.NUM_WORKERS = 2
cfg.TRAIN.PREFETCH_FACTOR = 4
cfg.TRAIN.FOLD = 1
cfg.TRAIN.LOAD_CHECKPOINT = True
cfg.TRAIN.SAVE_TOP_K = 5
cfg.TRAIN.IDX_CHECKPOINT = -1
cfg.TRAIN.WANDB = True
cfg.TRAIN.AUGMENTATION = True


cfg.TRAIN.PRETRAIN = "imagenet"  #  "pcl" or imagenet

# adjust input dim of the model
cfg.DATA.DIM_SIZE = 1

cfg.SYS.ACCELERATOR = "gpu"
cfg.SYS.DEVICES = [0]
cfg.SYS.MIX_PRECISION = "16-mixed"  # 32 or 16-mixed

cfg.OPT.LEARNING_RATE = 0.001
cfg.OPT.FACTOR_LR = 0.5
cfg.OPT.PATIENCE_LR = 50
cfg.OPT.PATIENCE_ES = 270


# cfg.PREDICT.IDX_CHECKPOINT = -1
cfg.PREDICT.BATCH_SIZE = 8
cfg.PREDICT.MIN_SIZE_REMOVE = 3
cfg.PREDICT.MODE = "2D"  # "3D"
cfg.PREDICT.ENSEMBLE = False
cfg.PREDICT.MASK_EXIST = False
cfg.PREDICT.MODEL = "tiramisu"  # "tiramisu"or resnet50
# cfg.PREDICT.NAME_ZIP = f"{cfg.PREDICT.MODEL}_95900195_{cfg.PREDICT.MIN_SIZE_REMOVE}" # or resnet50
cfg.PREDICT.NAME_ZIP = "vinbigdata_nhat"
cfg.PREDICT.WEIGHTS = [0.9, 0.84, 0.01, 1, 0.95]
# cfg.PREDICT.WEIGHTS = [0.9,0.84,0.01,1,0.95]
# 93.255
# 0.999452178396494	0.998410243189825	0.01 1	0.999097705594225


cfg.DIRS.SAVE_DIR = f"./weights_{cfg.TRAIN.TASK}_{cfg.PREDICT.MODEL}/"

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
cfg.TRAIN.NUM_WORKERS = 2
cfg.TRAIN.PREFETCH_FACTOR = 4

cfg.TRAIN.LOAD_CHECKPOINT = True
cfg.TRAIN.SAVE_TOP_K = 1
cfg.TRAIN.IDX_CHECKPOINT = -1
cfg.TRAIN.WANDB = True



cfg.PREDICT.BATCH_SIZE = 8
cfg.PREDICT.MIN_SIZE_REMOVE = 3

import torch
import numpy as np
from skimage import measure

# import OrderedDict
# import Pearson's correlation coefficient
from scipy.stats import pearsonr
from skimage import measure

from collections import OrderedDict


def dice_slice(y_true, y_pred, class_index=1, smooth=1e-5):
    output_standard = torch.argmax(y_pred, dim=1, keepdim=True)
    output = torch.where(output_standard == class_index, 1, 0)
    label = torch.where(y_true == class_index, 1, 0)

    intersection = torch.sum(label * output, dim=[1, 2, 3])
    union = torch.sum(label, dim=[1, 2, 3]) + torch.sum(output, dim=[1, 2, 3])
    return torch.mean((2.0 * intersection + smooth) / (union + smooth), dim=0)


def dice_volume(y_true, y_pred, class_index=1, smooth=1e-5):
    if class_index == 1:
        y_pred = np.where(y_pred == class_index, 1, 0)
        y_true = np.where(y_true == class_index, 1, 0)
    else:
        y_pred = np.where(y_pred >= class_index, 1, 0)
        y_true = np.where(y_true >= class_index, 1, 0)

    intersection = np.sum(y_true * y_pred)
    cardinality = np.sum(y_true + y_pred)
    return (2.0 * intersection + smooth) / (cardinality + smooth)

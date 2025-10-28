import torch
import numpy as np
from skimage import measure

# import OrderedDict
# import Pearson's correlation coefficient
from scipy.stats import pearsonr
from skimage import measure
from medpy.metric.binary import dc, hd
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

def dice_volume_ACDC(y_true, y_pred, class_index=1, smooth=1e-5):
    y_pred = np.where(y_pred == class_index, 1, 0)
    y_true = np.where(y_true == class_index, 1, 0)
    intersection = np.sum(y_true * y_pred)
    cardinality = np.sum(y_true + y_pred)
    return (2.0 * intersection + smooth) / (cardinality + smooth)

def metrics_ACDC(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    res = []
    # Loop on each classes of the input images
    for c in [3, 1, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)
        # compute the hd
        hd_value = hd(gt_c_i, pred_c_i, voxelspacing=voxel_size)
        # Compute volume
        vol_pred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        vol_gt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice, hd_value, vol_pred, abs(vol_pred-vol_gt)]

    return res

def metrics_EMIDEC(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    Return
    ------
    A list of metrics in this order,     
    "Dice_Myocardium","HD_Myocardium", "Volume_Myocardium", "Err_Myocardium(ml)",
    "Dice_Infarction", "Volume_Infarction", "Err_Infarction(ml)", Vol_Difference_Infarction_rate(%),
    "Dice_No-Reflow", "Volume_No-Reflow", "Err_No-Reflow(ml)", Vol_Difference_No-Reflow_rate(%)
    """
    res = []

    gt_myo = (img_gt == 2)+ (img_gt == 3) + (img_gt == 4)
    pred_myo = (img_pred == 2) + (img_pred == 3) + (img_pred == 4)
    dice_myo = dc(gt_myo, pred_myo)
    hd_myo = hd(gt_myo, pred_myo, voxelspacing=voxel_size)
    vol_pred_myo = pred_myo.sum() * np.prod(voxel_size) / 1000.
    vol_gt_myo = gt_myo.sum() * np.prod(voxel_size) / 1000.
    res += [dice_myo, hd_myo, vol_pred_myo, abs(vol_pred_myo-vol_gt_myo)]

    gt_infarction = (img_gt == 3) + (img_gt == 4)
    pred_infarction = (img_pred == 3) + (img_pred == 4)
    dice_infarction = dc(gt_infarction, pred_infarction)
    vol_pred_infarction = pred_infarction.sum() * np.prod(voxel_size) / 1000.
    vol_gt_infarction = gt_infarction.sum() * np.prod(voxel_size) / 1000.
    vol_difference_infarction_rate = abs(vol_pred_infarction-vol_gt_infarction) / vol_gt_myo * 100
    res += [dice_infarction, vol_pred_infarction, abs(vol_pred_infarction-vol_gt_infarction), vol_difference_infarction_rate]

    gt_noreflow = (img_gt == 4)
    pred_noreflow = (img_pred == 4)
    dice_noreflow = dc(gt_noreflow, pred_noreflow)
    vol_pred_noreflow = pred_noreflow.sum() * np.prod(voxel_size) / 1000.
    vol_gt_noreflow = gt_noreflow.sum() * np.prod(voxel_size) / 1000.
    vol_difference_noreflow_rate = abs(vol_pred_noreflow-vol_gt_noreflow) / vol_gt_myo * 100
    res += [dice_noreflow, vol_pred_noreflow, abs(vol_pred_noreflow-vol_gt_noreflow), vol_difference_noreflow_rate]



    return res
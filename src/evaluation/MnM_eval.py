import sys
sys.path.append("/home/nhattm/EMIDEC/")
import torch
import numpy as np
import nibabel as nib
from natsort import natsorted
import glob
from tqdm import tqdm
from segment2d import *
checkpoint = "weights_MnM_train_full/last.ckpt"
cfg.DATA.CLASS_WEIGHT = [1, 12, 12,12]
num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCDenseNet(in_channels=cfg.DATA.INDIM_MODEL, n_classes=num_classes)
segmenter = Segmenter_MnM(
    model,
    cfg.DATA.CLASS_WEIGHT,
    4,
    0.001,
    0.5,
    50,
)

segmenter = Segmenter_MnM.load_from_checkpoint(
    checkpoint_path=checkpoint,
    model=model,
    class_weight=cfg.DATA.CLASS_WEIGHT,
    num_classes=num_classes,
    learning_rate=0.001,
    factor_lr=0.5,
    patience_lr=50,
)
segmenter.eval()
segmenter = segmenter.to(device)

list_of_test = natsorted(glob.glob("MnM_data/Testing/*[!gt].nii.gz"))
dice_scores = {"dice_myo_ED":[], "dice_lv_ED":[], "dice_rv_ED":[],
               "dice_myo_ES":[], "dice_lv_ES":[], "dice_rv_ES":[]}

if __name__ == "__main__":
    for image_path in tqdm(list_of_test):
        id_patient = image_path.split("/")[-1].split("_")[0]
        image_nii = nib.load(image_path)
        mask_nii = nib.load(image_path.replace(".nii.gz", "_gt.nii.gz"))
        image = image_nii.get_fdata()
        mask = mask_nii.get_fdata().astype(np.uint8)


        data = preprocess_data(image_path)
        seg = predict_data_model(data, model).astype(np.uint8)


        dice_myo = dice_volume_MnM(mask, seg, class_index=2)
        dice_lv = dice_volume_MnM(mask, seg, class_index=1)
        dice_rv = dice_volume_MnM(mask, seg, class_index=3)

        if "ED" in image_path:
            dice_scores["dice_myo_ED"].append(dice_myo)
            dice_scores["dice_lv_ED"].append(dice_lv)
            dice_scores["dice_rv_ED"].append(dice_rv)
        else:
            dice_scores["dice_myo_ES"].append(dice_myo)
            dice_scores["dice_lv_ES"].append(dice_lv)
            dice_scores["dice_rv_ES"].append(dice_rv)

print("dice_myo_ED: ", np.mean(dice_scores["dice_myo_ED"]))
print("dice_lv_ED: ", np.mean(dice_scores["dice_lv_ED"]))
print("dice_rv_ED: ", np.mean(dice_scores["dice_rv_ED"]))
print("dice_myo_ES: ", np.mean(dice_scores["dice_myo_ES"]))
print("dice_lv_ES: ", np.mean(dice_scores["dice_lv_ES"]))
print("dice_rv_ES: ", np.mean(dice_scores["dice_rv_ES"]))

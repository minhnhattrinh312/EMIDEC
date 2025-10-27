import nibabel as nib
import glob
import os
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("./")
from segment2d import *

from natsort import natsorted
import csv
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dim2pad", type=list, default=[256, 256])
# parser.add_argument("--save_path", type=str, default="./emidec_train_full/")
parser.add_argument("--split_dataset", type=str, default="train_test_val", help="options are: train_test, train_test_val")
parser.add_argument("--task", type=str, default="train_full", help="options are: train_full, train_combine_myo")
parser.add_argument("--use_specific_pts", type=bool, default=False)
args = parser.parse_args()

list_of_train = natsorted(glob.glob("MnM_data/Training/*[!gt].nii.gz"))
list_of_val = natsorted(glob.glob("MnM_data/Validation/*[!gt].nii.gz"))
list_of_test = natsorted(glob.glob("MnM_data/Testing/*[!gt].nii.gz"))

random.seed(42)

if args.split_dataset == "train_test":
    list_train = list_of_train + list_of_val
    list_test = list_of_test
else:
    list_train = list_of_train
    list_val = list_of_val
    list_test = list_of_test


print(f"Number of train: {len(list_train)}")
print(f"Number of val: {len(list_val)}")
print(f"Number of test: {len(list_test)}")

if __name__ == "__main__":
    if args.task == "train_full":
        save_path = "./MnM_train_full/"
    elif args.task == "train_combine_myo":
        save_path = "./MnM_train_myo/"
    else:
        raise ValueError("Invalid value for task,")
    print(f"this task is for {args.task}")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("csv_files", exist_ok=True)
    for image_path in tqdm(list_train):
        id_patient = image_path.split("/")[-1].split("_")[0]
        image_nii = nib.load(image_path)
        mask_nii = nib.load(image_path.replace(".nii.gz", "_gt.nii.gz"))
        
        image = image_nii.get_fdata()
        mask = mask_nii.get_fdata().astype(np.uint8)

        combined_mask = mask.copy()
        combined_mask[mask == 3] = 0

        image = min_max_normalize(image)
        padded_image, crop_index, padded_index = pad_background(image, dim2pad=args.dim2pad, x_shift=-50)
        padded_mask = pad_background_with_index(mask, crop_index, padded_index, dim2pad=args.dim2pad)
        padded_combined_mask = pad_background_with_index(combined_mask, crop_index, padded_index, dim2pad=args.dim2pad)

        test_mask = invert_padding(image.shape, padded_mask, crop_index, padded_index)

        if np.linalg.norm(test_mask - mask) > 0:
            print(image_path)
        num_slices = 0
        for i in range(padded_image.shape[-1]):
            if np.sum(padded_mask[:, :, i]) == 0:
                continue
            num_slices += 1
            slice_image = padded_image[:, :, i : i + 1]
            if args.task == "train_full":
                slice_mask = padded_mask[:, :, i]
            else:
                slice_mask = padded_combined_mask[:, :, i]

            # if np.sum(slice_mask) - np.sum(mask[..., i]) != 0:
            #     raise ValueError("Error in padding")
            np.savez_compressed(
                os.path.join(save_path, f"{id_patient}_{i}.npz"),
                image=slice_image,
                mask=slice_mask.astype(np.uint8),
            )

    # create csv file to save id_patient, path of patient for val and test
    with open(f"csv_files/val_MnM_{args.task}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id_patient", "path"])
        for image_path in list_val:
            id_patient = image_path.split("/")[-2]
            writer.writerow([id_patient, image_path])

    with open(f"csv_files/test_MnM_{args.task}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id_patient", "path"])
        for image_path in list_test:
            id_patient = image_path.split("/")[-2]
            writer.writerow([id_patient, image_path])

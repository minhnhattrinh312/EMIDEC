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
parser.add_argument("--dim_resize", type=int, default=256)
# parser.add_argument("--save_path", type=str, default="./emidec_train_full/")
parser.add_argument("--split_dataset", type=str, default="train_test_val", help="options are: train_test, train_test_val")
parser.add_argument(
    "--task", type=str, default="train_full", help="options are: train_full, train_combine, train_combine_myo"
)
parser.add_argument("--use_specific_pts", type=bool, default=False)
args = parser.parse_args()

list_image_normal = natsorted(glob.glob("./emidec-dataset-1.0.1/Case_N*/I*/*"))
list_image_pathologic = natsorted(glob.glob("./emidec-dataset-1.0.1/Case_P*/I*/*"))
# split the dataset into training and validation and test with sample ratio 70:10:20

random.seed(42)
random.shuffle(list_image_normal)
random.shuffle(list_image_pathologic)
if args.split_dataset == "train_test_val":
    train_normal = list_image_normal[: int(0.7 * len(list_image_normal)) + 1]
    val_normal = list_image_normal[int(0.7 * len(list_image_normal)) + 1 : int(0.8 * len(list_image_normal)) + 1]
    test_normal = list_image_normal[int(0.8 * len(list_image_normal)) + 1 :]
    train_pathologic = list_image_pathologic[: int(0.7 * len(list_image_pathologic))]
    val_pathologic = list_image_pathologic[
        int(0.7 * len(list_image_pathologic)) : int(0.8 * len(list_image_pathologic))
    ]
    test_pathologic = list_image_pathologic[int(0.8 * len(list_image_pathologic)) :]

    list_train = train_normal + train_pathologic
    list_val = val_normal + val_pathologic
    list_test = test_normal + test_pathologic
    print(f"Number of train: {len(list_train)} with {len(train_normal)} normal and {len(train_pathologic)} pathologic")
    print(f"Number of val: {len(list_val)} with {len(val_normal)} normal and {len(val_pathologic)} pathologic")
    print(f"Number of test: {len(list_test)} with {len(test_normal)} normal and {len(test_pathologic)} pathologic")
elif args.split_dataset == "train_test":
    train_normal = list_image_normal[: int(0.8 * len(list_image_normal)) + 1]
    test_normal = list_image_normal[int(0.8 * len(list_image_normal)) + 1 :]
    train_pathologic = list_image_pathologic[: int(0.8 * len(list_image_pathologic))]
    test_pathologic = list_image_pathologic[int(0.8 * len(list_image_pathologic)) :]

    list_train = train_normal + train_pathologic
    list_val = []
    list_test = test_normal + test_pathologic
else:
    raise ValueError("Invalid value for split_dataset")

MI_test_pts = [
    "Case_P050",
    "Case_P087",
    "Case_P001",
    "Case_P010",
    "Case_P017",
    "Case_P029",
    "Case_P090",
    "Case_P038",
    "Case_N052",
    "Case_N016",
    "Case_P100",
    "Case_P043",
    "Case_P051",
    "Case_N030",
    "Case_P007",
    "Case_P088",
    "Case_N025",
    "Case_P076",
    "Case_N046",
    "Case_N054",
    "Case_N049",
    "Case_N041",
    "Case_N023",
    "Case_P026",
    "Case_P031",
    "Case_N024",
    "Case_P064",
    "Case_P021",
    "Case_P015",
    "Case_P094",
]
if args.use_specific_pts:
    list_test = [f"./emidec-dataset-1.0.1/{patiend_id}/Images/{patiend_id}.nii.gz" for patiend_id in MI_test_pts]
    # list train will be the rest of the dataset
    list_train = list(set(list_image_normal + list_image_pathologic) - set(list_test))

print(f"Number of train: {len(list_train)}")

if __name__ == "__main__":
    if args.task == "train_full":
        save_path = "./emidec_train_full/"
    elif args.task == "train_combine":
        save_path = "./emidec_train_combine/"
    elif args.task == "train_combine_myo":
        save_path = "./emidec_train_combine_myo/"
    else:
        raise ValueError("Invalid value for task,")
    print(f"this task is for {args.task}")
    os.makedirs(save_path, exist_ok=True)
    for image_path in tqdm(list_train):
        id_patient = image_path.split("/")[-3]
        image = nib.load(image_path).get_fdata()
        mask = nib.load(image_path.replace("Images", "Contours")).get_fdata()

        combined_mask = mask.copy()
        combined_mask[mask == 4] = 3

        combined_myo = combined_mask.copy()
        combined_myo[combined_mask == 3] = 2
        image = min_max_normalize(image)
        resize_image, restore_info = crop_resize_image(image, new_dim=args.dim_resize)
        resize_mask = crop_resize_mask(mask, restore_info)
        resize_combined_mask = crop_resize_mask(combined_mask, restore_info)
        resize_combined_myo = crop_resize_mask(combined_myo, restore_info)

        for i in range(resize_image.shape[-1]):
            if np.sum(resize_mask[:, :, i]) == 0:
                continue
            slice_image = resize_image[:, :, i : i + 1]
            if args.task == "train_full":
                slice_mask = resize_mask[:, :, i]
            elif args.task == "train_combine":
                slice_mask = resize_combined_mask[:, :, i]
            elif args.task == "train_combine_myo":
                slice_mask = resize_combined_myo[:, :, i]

            # if np.sum(slice_mask) - np.sum(mask[..., i]) != 0:
            #     raise ValueError("Error in padding")
            np.savez_compressed(
                os.path.join(save_path, f"{id_patient}_{i}.npz"),
                image=slice_image,
                mask=slice_mask.astype(np.uint8),
            )

    # create csv file to save id_patient, path of patient for val and test
    with open(f"csv_files/EMIDEC_val_{args.task}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id_patient", "path"])
        for image_path in list_val:
            id_patient = image_path.split("/")[-3]
            image = nib.load(image_path).get_fdata()
            writer.writerow([id_patient, image_path])

    with open(f"csv_files/EMIDEC_test_{args.task}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id_patient", "path"])
        for image_path in list_test:
            id_patient = image_path.split("/")[-3]
            image = nib.load(image_path).get_fdata()
            writer.writerow([id_patient, image_path])

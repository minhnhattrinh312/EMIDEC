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
parser.add_argument("--dim2pad", type=list, default=[128, 128])
parser.add_argument("--save_path", type=str, default="./emidec_train/")
parser.add_argument("--spilt_dataset", type=str, default="train_test", help="options are: train_test, train_test_val")
args = parser.parse_args()

list_image_normal = natsorted(glob.glob("./emidec-dataset-1.0.1/Case_N*/I*/*"))
list_image_pathologic = natsorted(glob.glob("./emidec-dataset-1.0.1/Case_P*/I*/*"))
# split the dataset into training and validation and test with sample ratio 70:10:20
random.seed(42)
random.shuffle(list_image_normal)
random.shuffle(list_image_pathologic)
if args.spilt_dataset == "train_test_val":
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

elif args.spilt_dataset == "train_test":
    train_normal = list_image_normal[: int(0.8 * len(list_image_normal)) + 1]
    test_normal = list_image_normal[int(0.8 * len(list_image_normal)) + 1 :]
    train_pathologic = list_image_pathologic[: int(0.8 * len(list_image_pathologic))]
    test_pathologic = list_image_pathologic[int(0.8 * len(list_image_pathologic)) :]

    list_train = train_normal + train_pathologic
    list_val = []
    list_test = test_normal + test_pathologic
else:
    raise ValueError("Invalid value for spilt_dataset")

if __name__ == "__main__":
    os.makedirs(args.save_path, exist_ok=True)
    for image_path in tqdm(list_train):
        id_patient = image_path.split("/")[-3]
        image = nib.load(image_path).get_fdata()
        mask = nib.load(image_path.replace("Images", "Contours")).get_fdata()
        image = min_max_normalize(image)
        padded_image, crop_index, padded_index = pad_background(image, dim2pad=args.dim2pad)
        padded_mask = pad_background_with_index(mask, crop_index, padded_index, dim2pad=args.dim2pad)

        for i in range(padded_image.shape[-1]):
            slice_image = padded_image[:, :, i : i + 1]
            slice_mask = padded_mask[:, :, i]
            np.savez_compressed(
                os.path.join(args.save_path, f"{id_patient}_{i}.npz"),
                image=slice_image,
                mask=slice_mask.astype(np.uint8),
            )

    # create csv file to save id_patient, path of patient for val and test
    with open("val.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id_patient", "path", "num_slices"])
        for image_path in list_val:
            id_patient = image_path.split("/")[-3]
            image = nib.load(image_path).get_fdata()
            writer.writerow([id_patient, image_path, image.shape[-1]])

    with open("test.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id_patient", "path", "num_slices"])
        for image_path in list_test:
            id_patient = image_path.split("/")[-3]
            image = nib.load(image_path).get_fdata()
            writer.writerow([id_patient, image_path, image.shape[-1]])

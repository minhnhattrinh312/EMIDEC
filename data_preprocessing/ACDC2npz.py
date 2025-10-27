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
import os
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--dim_resize", type=int, default=256)
parser.add_argument("--save_path", type=str, default="ACDC_train/")

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
list_patient_train_val = natsorted(glob.glob("ACDC/database/training/patient*"))
list_patient_test = natsorted(glob.glob("ACDC/database/testing/patient*"))

# from 1-20 is group DCM, 21-40 is group HCM, 41-60 is group MINF, 61-80 is group NOR, 81-100 is group ARV
list_patient_train_val_dcm = list_patient_train_val[:20]
list_patient_train_val_hcm = list_patient_train_val[20:40]
list_patient_train_val_minf = list_patient_train_val[40:60]
list_patient_train_val_nor = list_patient_train_val[60:80]
list_patient_train_val_arv = list_patient_train_val[80:100]

# take random 80% of each group for training and 20% for validation with specific seed 42
random.seed(42)
list_patient_train_dcm = random.sample(list_patient_train_val_dcm, int(0.8 * len(list_patient_train_val_dcm)))
list_patient_train_hcm = random.sample(list_patient_train_val_hcm, int(0.8 * len(list_patient_train_val_hcm)))
list_patient_train_minf = random.sample(list_patient_train_val_minf, int(0.8 * len(list_patient_train_val_minf)))
list_patient_train_nor = random.sample(list_patient_train_val_nor, int(0.8 * len(list_patient_train_val_nor)))
list_patient_train_arv = random.sample(list_patient_train_val_arv, int(0.8 * len(list_patient_train_val_arv)))
list_patient_val_dcm = list(set(list_patient_train_val_dcm) - set(list_patient_train_dcm))
list_patient_val_hcm = list(set(list_patient_train_val_hcm) - set(list_patient_train_hcm))
list_patient_val_minf = list(set(list_patient_train_val_minf) - set(list_patient_train_minf))
list_patient_val_nor = list(set(list_patient_train_val_nor) - set(list_patient_train_nor))
list_patient_val_arv = list(set(list_patient_train_val_arv) - set(list_patient_train_arv))

list_patient_train = list_patient_train_dcm + list_patient_train_hcm + list_patient_train_minf + list_patient_train_nor + list_patient_train_arv
list_patient_val = list_patient_val_dcm + list_patient_val_hcm + list_patient_val_minf + list_patient_val_nor + list_patient_val_arv
print(f"Number of train: {len(list_patient_train)}")
print(f"Number of val: {len(list_patient_val)}")
print(f"Number of test: {len(list_patient_test)}")

count = 0
for patient_path in tqdm(list_patient_train):
    id_patient = patient_path.split("/")[-1]
    slice_count = 0
    for gt_img_path in glob.glob(patient_path + "/*gt.nii.gz"):
        img_path = gt_img_path.replace("_gt", "")
        image = nib.load(img_path).get_fdata()
        mask = nib.load(gt_img_path).get_fdata()
        image = min_max_normalize(image)
        resize_image, restore_info = crop_resize_image(image)
        resize_mask = crop_resize_mask(mask, restore_info)
        for i in range(resize_image.shape[-1]):
            if np.sum(resize_mask[:, :, i]) == 0:
                continue
            slice_image = resize_image[:, :, i : i + 1]
            slice_mask = resize_mask[:, :, i]
            slice_count += 1
            count += 1
            np.savez_compressed(os.path.join(args.save_path, f"{id_patient}_{slice_count}.npz"), image=slice_image, mask=slice_mask.astype(np.uint8))
print(f"Number of slices: {count}")

# create csv file to save id_patient, path of patient for val and test
with open(f"csv_files/ACDC_val.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id_patient", "path"])
    for patient_path in list_patient_val:
        id_patient = patient_path.split("/")[-1]
        for gt_img_path in glob.glob(patient_path + "/*gt.nii.gz"):
            image_path = gt_img_path.replace("_gt", "")
            image = nib.load(image_path).get_fdata()
            writer.writerow([id_patient, image_path])


with open(f"csv_files/ACDC_test.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id_patient", "path"])
    for patient_path in list_patient_test:
        id_patient = patient_path.split("/")[-1]
        for gt_img_path in glob.glob(patient_path + "/*gt.nii.gz"):
            image_path = gt_img_path.replace("_gt", "")
            image = nib.load(image_path).get_fdata()
            writer.writerow([id_patient, image_path])

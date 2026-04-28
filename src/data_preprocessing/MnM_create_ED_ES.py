import nibabel as nib
import glob
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from natsort import natsorted
import shutil

#download opendataset from https://www.ub.edu/mnms/
#download info_data from https://www.ub.edu/mnms/

list_of_train = natsorted(glob.glob("OpenDataset/Training/*/*/*sa.nii.gz"))
list_of_val = natsorted(glob.glob("OpenDataset/Validation/*/*sa.nii.gz"))
list_of_test = natsorted(glob.glob("OpenDataset/Testing/*/*sa.nii.gz"))
shutil.copy("OpenDataset/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv", "MnM_data/")

os.makedirs("MnM_data/Training/", exist_ok=True)
os.makedirs("MnM_data/Validation/", exist_ok=True)
os.makedirs("MnM_data/Testing/", exist_ok=True)
# copy info_data to MnM_data
info_data = pd.read_csv("MnM_data/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")

dict_of_data = {"Training": list_of_train, "Validation": list_of_val, "Testing": list_of_test}
for key, list_of_data in dict_of_data.items():
    print(f"Processing {key} data")
    for image_path in tqdm(list_of_data):
        image_nii = nib.load(image_path)
        mask_nii = nib.load(image_path.replace("sa.nii.gz", "sa_gt.nii.gz"))
        image = image_nii.get_fdata()
        mask = mask_nii.get_fdata().astype(np.uint8)
        patient_id = image_path.split("/")[-1].split("_")[0]
        # get ED and ES from info_data
        ED_frame = info_data[info_data["External code"] == patient_id]["ED"].values[0]
        ES_frame = info_data[info_data["External code"] == patient_id]["ES"].values[0]
        
        image_ED = image[..., ED_frame]
        image_ES = image[..., ES_frame]
        mask_ED = mask[..., ED_frame]
        mask_ES = mask[..., ES_frame]

        # save image_ED and mask_ED with the same space and resolution as the original image
        nib.save(nib.Nifti1Image(image_ED, image_nii.affine, header=image_nii.header), f"MnM_data/{key}/{patient_id}_ED.nii.gz")
        nib.save(nib.Nifti1Image(mask_ED, mask_nii.affine, header=mask_nii.header), f"MnM_data/{key}/{patient_id}_ED_gt.nii.gz")
        nib.save(nib.Nifti1Image(image_ES, image_nii.affine, header=image_nii.header), f"MnM_data/{key}/{patient_id}_ES.nii.gz")
        nib.save(nib.Nifti1Image(mask_ES, mask_nii.affine, header=mask_nii.header), f"MnM_data/{key}/{patient_id}_ES_gt.nii.gz")

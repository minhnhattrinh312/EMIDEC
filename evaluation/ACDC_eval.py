
import numpy as np
import csv
from natsort import natsorted
import configparser
import os
import sys
import torch
sys.path.append("./")
import argparse
from segment2d import *
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_acdc = torch.load("tiramisu_acdc.pt", weights_only=False)
model_acdc.eval()
model_acdc = model_acdc.to(device)

metrics_acdc_header = ["patient_name", 
            "Dice ED Left Ventricle", "HD ED Left Ventricle", "Volume ED Left Ventricle", "Err ED Left Ventricle(ml)",
          "Dice ED Right Ventricle", "HD ED Right Ventricle", "Volume ED Right Ventricle", "Err ED Right Ventricle(ml)",
          "Dice ED Myocardium", "HD ED Myocardium", "Volume ED Myocardium", "Err ED Myocardium(ml)",
          "Dice ES Left Ventricle", "HD ES Left Ventricle", "Volume ES Left Ventricle", "Err ES Left Ventricle(ml)",
          "Dice ES Right Ventricle", "HD ES Right Ventricle", "Volume ES Right Ventricle", "Err ES Right Ventricle(ml)",
          "Dice ES Myocardium", "HD ES Myocardium", "Volume ES Myocardium", "Err ES Myocardium(ml)"]

config = configparser.ConfigParser()
with open("csv_files/ACDC_test.csv", mode="r") as f:
    reader = csv.DictReader(f)
    list_test_subject = [row["path"] for row in reader]

test_patitent_paths = natsorted(glob.glob("ACDC/database/testing/patient*"))

def evaluate_acdc(csv_file_name="result_acdc.csv", save_nii_files=True):
    import time 
    total_time_inference = []
    # open csv file to save the result
    f = open(f"csv_files/{csv_file_name}", mode="w")
    writer = csv.DictWriter(f, fieldnames=metrics_acdc_header)
    writer.writeheader()
    # create folder to save the predicted mask
    os.makedirs("predicted_acdc", exist_ok=True)
    for patient_path in tqdm(test_patitent_paths):
        
        info_path = patient_path + "/Info.cfg"
        with open(info_path, mode="r") as f:
            config.read_string(f"[info]\n{f.read()}")
        ED_index = f"0{config['info']['ED']}" if int(config['info']['ED']) < 10 else f"{config['info']['ED']}"
        ES_index = f"0{config['info']['ES']}" if int(config['info']['ES']) < 10 else f"{config['info']['ES']}"
        patient_name = patient_path.split("/")[-1]
        ED_image_path = patient_path + f"/{patient_name}_frame{ED_index}.nii.gz"
        ES_image_path = patient_path + f"/{patient_name}_frame{ES_index}.nii.gz"
        ED_mask_path = patient_path + f"/{patient_name}_frame{ED_index}_gt.nii.gz"
        ES_mask_path = patient_path + f"/{patient_name}_frame{ES_index}_gt.nii.gz"


        _, ED_affine, ED_header = load_nii(ED_image_path)
        _, ES_affine, ES_header = load_nii(ES_image_path)
        ED_mask, _, _ = load_nii(ED_mask_path)
        ES_mask, _, _ = load_nii(ES_mask_path)

        ED_data = preprocess_data_nii(ED_image_path)
        ES_data = preprocess_data_nii(ES_image_path)
        start_time = time.time()
        ED_seg = predict_data_model(ED_data, model_acdc, min_size_remove=800, device=device).astype(np.uint8)
        total_time_inference.append(time.time() - start_time)
        start_time = time.time()
        ES_seg = predict_data_model(ES_data, model_acdc, min_size_remove=800, device=device).astype(np.uint8)
        total_time_inference.append(time.time() - start_time)
        result_ED = metrics_ACDC(ED_mask, ED_seg, voxel_size=ED_header.get_zooms())
        result_ES = metrics_ACDC(ES_mask, ES_seg, voxel_size=ES_header.get_zooms())
        # round the result to 3 decimal places
        result = [patient_name] + [round(r, 4) for r in result_ED] + [round(r, 4) for r in result_ES]

        # write the result to the csv file
        writer.writerow(dict(zip(metrics_acdc_header, result)))
        # save the predicted mask
        if save_nii_files:
            save_nii(f"predicted_acdc/{patient_name}_tiramisu_seg_ED.nii.gz", ED_seg, ED_affine, ED_header)
            save_nii(f"predicted_acdc/{patient_name}_tiramisu_seg_ES.nii.gz", ES_seg, ES_affine, ES_header)
        # break

    f.close()

    print("Total time inference:")
    for i, time in enumerate(total_time_inference):
        print(f"Patient {list_test_subject[i]}: {time} seconds")
    print(f"Total time inference: {sum(total_time_inference)} seconds")
    print(f"Average time inference: {sum(total_time_inference) / len(total_time_inference)} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_name", type=str, default="result_acdc.csv")
    parser.add_argument("--save_nii_files", type=bool, default=True)

    args = parser.parse_args()
    evaluate_acdc(csv_file_name=args.csv_file_name, save_nii_files=args.save_nii_files)
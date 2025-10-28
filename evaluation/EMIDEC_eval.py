
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
model_emidec = torch.load("tiramisu_emidec.pt", weights_only=False)
model_emidec.eval()
model_emidec = model_emidec.to(device)

def evaluate_emidec(csv_file_name="result_emidec.csv", save_nii_files=True):
    import time
    total_time_inference = []
    metrics_emidec_header = ["patient_name", 
        "Dice_Myocardium","HD_Myocardium", "Volume_Myocardium", "Err_Myocardium(ml)",
        "Dice_Infarction", "Volume_Infarction", "Err_Infarction(ml)", "Vol_Difference_Infarction_rate(%)",
        "Dice_No-Reflow", "Volume_No-Reflow", "Err_No-Reflow(ml)", "Vol_Difference_No-Reflow_rate(%)",
        ]
    # create folder to save the predicted image
    os.makedirs("predicted_emidec", exist_ok=True)

    with open("csv_files/EMIDEC_test_train_full.csv", mode="r") as f:
        reader = csv.DictReader(f)
        list_test_subject = [row["path"] for row in reader]

    # open csv file to save the result
    f = open(f"csv_files/{csv_file_name}", mode="w")
    writer = csv.DictWriter(f, fieldnames=metrics_emidec_header)
    writer.writeheader()


    for image_path in tqdm(list_test_subject):

        patient_name = image_path.split("/")[-3]
        # print("patient name: ", patient_name)
        mask_path = image_path.replace("Images", "Contours")
        image, affine, header = load_nii(image_path)
        mask, _, _ = load_nii(mask_path)

        data = preprocess_data_nii(image_path)
        start_time = time.time()
        seg = predict_data_model_emidec(data, model_emidec, min_size_remove=500, device=device).astype(np.uint8)
        total_time_inference.append(time.time() - start_time)
        result = metrics_EMIDEC(mask, seg, voxel_size=header.get_zooms())
        # round the result to 3 decimal places
        result = [patient_name] + [round(r, 4) for r in result]
        # write the result to the csv file
        writer.writerow(dict(zip(metrics_emidec_header, result)))

        if save_nii_files:
            save_nii(f"predicted_emidec/{patient_name}_tiramisu_seg.nii.gz", seg, affine, header)

    f.close()
    print("Total time inference:")
    for i, time in enumerate(total_time_inference):
        print(f"Patient {list_test_subject[i]}: {time} seconds")
    print(f"Total time inference: {sum(total_time_inference)} seconds")
    print(f"Average time inference: {sum(total_time_inference) / len(total_time_inference)} seconds")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_name", type=str, default="result_emidec.csv")
    parser.add_argument("--save_nii_files", type=bool, default=True)

    args = parser.parse_args()
    evaluate_emidec(csv_file_name=args.csv_file_name, save_nii_files=args.save_nii_files)

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
import pynvml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_acdc = torch.load("tiramisu_acdc.pt", weights_only=False, map_location=device)
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

test_patient_paths = natsorted(glob.glob("ACDC/database/testing/patient*"))

def evaluate_acdc(csv_file_name="result_acdc.csv", save_nii_files=True):
    import time 
    total_time_inference = []
    # open csv file to save the result
    f = open(f"csv_files/{csv_file_name}", mode="w")
    writer = csv.DictWriter(f, fieldnames=metrics_acdc_header)
    writer.writeheader()
    # create folder to save the predicted mask
    os.makedirs("predicted_acdc", exist_ok=True)
    # Initialize NVML for power monitoring
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use GPU 0; adjust if needed
    for patient_path in tqdm(test_patient_paths):
        
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
        print(f"Patient {test_patient_paths[i]}: {time} seconds")
    print(f"Total time inference: {sum(total_time_inference)} seconds")
    print(f"Average time inference: {sum(total_time_inference) / len(total_time_inference)} seconds")

def evaluate_acdc_GPU(csv_file_name="result_acdc.csv", save_nii_files=True):
    """
    Evaluate ACDC dataset with energy consumption tracking using NVIDIA GPU.
    """
    import time 
    total_time_inference = []
    total_energy_joules = []  # To store energy per inference
    # Open CSV file to save the result
    f = open(f"csv_files/{csv_file_name}", mode="w")
    writer = csv.DictWriter(f, fieldnames=metrics_acdc_header)
    writer.writeheader()
    # Create folder to save the predicted mask
    os.makedirs("predicted_acdc", exist_ok=True)
    # Initialize NVML for power monitoring
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use GPU 0; adjust if needed

    for patient_path in tqdm(test_patient_paths):  # Corrected typo 'test_patitent_paths'
        info_path = patient_path + "/Info.cfg"
        with open(info_path, mode="r") as f:
            config = configparser.ConfigParser()
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

        # Energy measurement for ED inference
        power_samples_ED = []
        torch.cuda.synchronize()  # Sync before inference
        start_time = time.time()
        with torch.no_grad():
            ED_seg = predict_data_model(ED_data, model_acdc, min_size_remove=800, device=device).astype(np.uint8)
        torch.cuda.synchronize()  # Sync after inference
        end_time = time.time()
        inference_time_ED = end_time - start_time
        total_time_inference.append(inference_time_ED)

        # Sample power during ED inference (10ms interval)
        sample_time = start_time
        while sample_time < end_time:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_samples_ED.append(power_mw / 1000.0)  # Convert to Watts
            time.sleep(0.01)  # 10ms sampling
            sample_time += 0.01
        avg_power_ED = sum(power_samples_ED) / len(power_samples_ED) if power_samples_ED else 0.0
        energy_joules_ED = avg_power_ED * inference_time_ED

        # Energy measurement for ES inference
        power_samples_ES = []
        torch.cuda.synchronize()  # Sync before inference
        start_time = time.time()
        with torch.no_grad():
            ES_seg = predict_data_model(ES_data, model_acdc, min_size_remove=800, device=device).astype(np.uint8)
        torch.cuda.synchronize()  # Sync after inference
        end_time = time.time()
        inference_time_ES = end_time - start_time
        total_time_inference.append(inference_time_ES)

        # Sample power during ES inference (10ms interval)
        sample_time = start_time
        while sample_time < end_time:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_samples_ES.append(power_mw / 1000.0)  # Convert to Watts
            time.sleep(0.01)  # 10ms sampling
            sample_time += 0.01
        avg_power_ES = sum(power_samples_ES) / len(power_samples_ES) if power_samples_ES else 0.0
        energy_joules_ES = avg_power_ES * inference_time_ES

        # Total energy for this patient (ED + ES)
        energy_joules = energy_joules_ED + energy_joules_ES
        total_energy_joules.append(energy_joules)

        result_ED = metrics_ACDC(ED_mask, ED_seg, voxel_size=ED_header.get_zooms())
        result_ES = metrics_ACDC(ES_mask, ES_seg, voxel_size=ES_header.get_zooms())
        # Round the result to 3 decimal places
        result = [patient_name] + [round(r, 4) for r in result_ED] + [round(r, 4) for r in result_ES]

        # Write the result to the CSV file
        writer.writerow(dict(zip(metrics_acdc_header, result)))
        # Save the predicted mask
        if save_nii_files:
            save_nii(f"predicted_acdc/{patient_name}_tiramisu_seg_ED.nii.gz", ED_seg, ED_affine, ED_header)
            save_nii(f"predicted_acdc/{patient_name}_tiramisu_seg_ES.nii.gz", ES_seg, ES_affine, ES_header)

    f.close()

    # Cleanup NVML
    pynvml.nvmlShutdown()

    # Summary
    print("Total time inference:")
    for i, time in enumerate(total_time_inference):
        patient_idx = i // 2  # Since we have two inferences (ED, ES) per patient
        print(f"Patient {test_patient_paths[patient_idx]} - Phase {(i % 2 == 0 and 'ED' or 'ES')}: {time:.4f} seconds")
    print(f"Total time inference: {sum(total_time_inference):.4f} seconds")
    print(f"Average time inference per phase: {sum(total_time_inference) / len(total_time_inference):.4f} seconds")
    print(f"Total energy consumption: {sum(total_energy_joules):.4f} J")
    print(f"Average energy per patient (ED+ES): {sum(total_energy_joules) / len(total_energy_joules):.4f} J")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_name", type=str, default="result_acdc.csv")
    parser.add_argument("--save_nii_files", type=bool, default=True)

    args = parser.parse_args()
    # evaluate_acdc(csv_file_name=args.csv_file_name, save_nii_files=args.save_nii_files)
    evaluate_acdc_GPU(csv_file_name=args.csv_file_name, save_nii_files=args.save_nii_files)
                                                   
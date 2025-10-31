# import canvas
from reportlab.pdfgen import canvas
# import blue color
from reportlab.lib.colors import *
from reportlab.lib.utils import ImageReader
import io
import argparse
import numpy as np
import matplotlib.pyplot as plt
import configparser
import time
import sys
sys.path.append("./")
from segment2d import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_acdc = torch.load("tiramisu_acdc.pt", weights_only=False, map_location=device)
model_acdc.eval()
model_acdc = model_acdc.to(device)

def read_patient_acdc(info_path):
    patient_data = {}
    config = configparser.ConfigParser()
    with open(info_path, mode="r") as f:
        config.read_string(f"[info]\n{f.read()}")

    patient_data["Case"] = info_path.split("/")[-2]
    if config["info"]["Group"] == "DCM":
        patient_data["Group"] = "Dilated Cardiomyopathy"
    elif config["info"]["Group"] == "HCM":
        patient_data["Group"] = "Hypertrophic Cardiomyopathy"
    elif config["info"]["Group"] == "NOR":
        patient_data["Group"] = "Normal"
    elif config["info"]["Group"] == "RV":
        patient_data["Group"] = "Abnormal Right Ventricle"
    elif config["info"]["Group"] == "MINF":
        patient_data["Group"] = "Systolic Heart Failure with Infarction"
    patient_data["Height"] = config["info"]["Height"] + " cm"
    patient_data["Weight"] = config["info"]["Weight"] + " kg"
    patient_data["NbFrame"] = "Frame " + config["info"]["NbFrame"]
    patient_data["ED"] = f"0{config['info']['ED']}" if int(config['info']['ED']) < 10 else f"{config['info']['ED']}"
    patient_data["ES"] = f"0{config['info']['ES']}" if int(config['info']['ES']) < 10 else f"{config['info']['ES']}"
    patient_data["End-Diastolic (ED)"] = "Frame " + patient_data["ED"]
    patient_data["End-Systolic (ES)"] = "Frame " + patient_data["ES"]
    return patient_data


def create_acdc_report(pdf_file, info_path, segmentation_data_path=None, model=model_acdc):
    print("Creating report for patient with clinical data path: ", info_path)
    print("Creating report for patient with segmentation data path: ", segmentation_data_path)

    # --- Load data ---
    patient_dict = read_patient_acdc(info_path)
    patient_name = segmentation_data_path.split("/")[-2]
    # print(patient_name)
    ED_image_path = segmentation_data_path + f"/{patient_name}_frame{patient_dict['ED']}.nii.gz"
    ES_image_path = segmentation_data_path + f"/{patient_name}_frame{patient_dict['ES']}.nii.gz"
    #remove the key "ED" and "ES" from the patient_dict
    patient_dict.pop("ED")
    patient_dict.pop("ES")
    ED_image, _, ED_header = load_nii(ED_image_path)
    ES_image, _, ES_header = load_nii(ES_image_path)
    ED_image, ED_affine, ED_header = load_nii(ED_image_path)
    ES_image, ES_affine, ES_header = load_nii(ES_image_path)
    print("Predicting segmentation data...")
    # calculate the time of predicting segmentation data
    # image, restore_info = crop_resize_image(image, 256)
    data_ED = preprocess_data_nii(ED_image_path)
    data_ES = preprocess_data_nii(ES_image_path)
    start_time = time.time()
    seg_ED = predict_data_model(data_ED, model, min_size_remove=800, device=device).astype(np.uint8)
    print(f"Time taken to predict segmentation data ED: {time.time() - start_time} seconds")
    start_time = time.time()
    seg_ES = predict_data_model(data_ES, model, min_size_remove=800, device=device).astype(np.uint8)
    print(f"Time taken to predict segmentation data ES: {time.time() - start_time} seconds")
    # seg = crop_resize_mask(seg, restore_info)

    myocardium_volume_ED = round(make_volume(seg_ED == 2, ED_header.get_zooms()) / 1000, 2)
    right_ventricle_volume_ED = round(make_volume(seg_ED == 1, ED_header.get_zooms()) / 1000, 2)
    left_ventricle_volume_ED = round(make_volume(seg_ED == 3, ED_header.get_zooms()) / 1000, 2)
    myocardium_volume_ES = round(make_volume(seg_ES == 2, ES_header.get_zooms()) / 1000, 2)
    right_ventricle_volume_ES = round(make_volume(seg_ES == 1, ES_header.get_zooms()) / 1000, 2)
    left_ventricle_volume_ES = round(make_volume(seg_ES == 3, ES_header.get_zooms()) / 1000, 2)
    # calculate the ejection fraction
    lv_ef = round((left_ventricle_volume_ED - left_ventricle_volume_ES) / left_ventricle_volume_ED * 100, 2)
    rv_ef = round((right_ventricle_volume_ED - right_ventricle_volume_ES) / right_ventricle_volume_ED * 100, 2)
    myocardium_mass_ED = round(myocardium_volume_ED * 1.05, 2)
    myocardium_mass_ES = round(myocardium_volume_ES * 1.05, 2)
    # center crop the image and seg
    w, h = ED_image.shape[:2]
    ED_image = ED_image[w//2-64:w//2+64, h//2-64:h//2+64, :]
    seg_ED = seg_ED[w//2-64:w//2+64, h//2-64:h//2+64, :]

    # --- Choose middle slices ---
    z_slices = [1,seg_ED.shape[-1]//2,-2]

    # --- Create PDF canvas ---
    pagesize = (600, 600)
    c = canvas.Canvas(pdf_file, pagesize=pagesize)
    width, height = pagesize

    # --- Header ---
    c.drawImage('figures/logo1.png', 5, height - 80, width=80, height=80)
    c.setFont("Helvetica-Bold", 30)
    # make it center horizontally
    c.drawCentredString(width / 2, height - 50, "PATIENT REPORT")

    # --- Patient info ---
    infor_list = ["Case"]
    infor_x = 50
    for infor in infor_list:
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(blue)
        c.drawString(infor_x, height - 100, f"{infor}")
        c.setFillColor(black)
        c.setFont("Helvetica", 12)
        c.drawString(infor_x, height - 120, str(patient_dict.get(infor, "Data not available")))
        infor_x += 160

    # --- Create combined figure with 2 subplots ---
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    for i, z in enumerate(z_slices):
        ax[i].imshow(ED_image[..., z], cmap="gray", alpha=0.7)
        ax[i].imshow(seg_ED[..., z], cmap="jet", alpha=0.5)
        ax[i].axis("off")
        # adjust the space between the images   
    plt.subplots_adjust(wspace=0.05)

    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # --- Draw combined image on PDF ---
    image_x = 20
    image_y = height - 280
    image_w = 450
    image_h = 150
    c.drawImage(ImageReader(buf), image_x, image_y, width=image_w, height=image_h)
    # delete the buffer
    del buf
    # --- Add legend to the right of the image ---
    legend_x = image_x + image_w + 10
    legend_y = image_y + image_h / 2 - 5

    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#a36464"))
    c.rect(legend_x, legend_y, 10, 10, fill=1, stroke=0)
    c.setFillColor(HexColor("#c2b543"))
    c.rect(legend_x, legend_y - 20, 10, 10, fill=1, stroke=0)
    c.setFillColor(HexColor("#00AEEF"))
    c.rect(legend_x, legend_y - 40, 10, 10, fill=1, stroke=0)
    c.setFillColor(black)
    c.drawString(legend_x + 15, legend_y + 1, "Left Ventricle (LV)")
    c.drawString(legend_x + 15, legend_y - 19, "Myocardium (MYO)")
    c.drawString(legend_x + 15, legend_y - 39, "Right Ventricle (RV)")


    # --- Clinical data section ---
    col_width = width / 2
    clinical_y = height - 300
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(red)
    c.drawCentredString(col_width / 2, clinical_y, "Clinical Data")
    clinical_y -= 20
    c.setFillColor(black)
    c.setFont("Helvetica", 12)
    list_infor_remove = ["Case", "Sex", "Age"]
    for key, value in patient_dict.items():
        if key not in list_infor_remove:
            c.drawString(50, clinical_y, f"{key}: {value}")
            clinical_y -= 20

    # --- Segmentation data section ---
    segment_y = height - 300
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(red)
    c.drawCentredString(col_width + (col_width / 2), segment_y, "Segmentation Data")
    segment_y -= 20
    c.setFont("Helvetica", 12)
    segment_data = {
        "Volume": {
            "LV ED": f"{left_ventricle_volume_ED} mL",
            "LV ES": f"{left_ventricle_volume_ES} mL",
            "RV ED": f"{right_ventricle_volume_ED} mL",
            "RV ES": f"{right_ventricle_volume_ES} mL",
            "MYO ED": f"{myocardium_volume_ED} mL",
            "MYO ES": f"{myocardium_volume_ES} mL",
        },
        "Ejection Fraction": {
            "LV": f"{lv_ef}%",
            "RV": f"{rv_ef}%",
        },
        "Myocardium Mass": {
            "MYO Mass ED": f"{myocardium_mass_ED} g",
            "MYO Mass ES": f"{myocardium_mass_ES} g",
        }
    }
    c.setFillColor(black)
    for category, details in segment_data.items():
        c.drawString(col_width + 10, segment_y, f"{category}:")
        segment_y -= 20
        for sub_key, sub_value in details.items():
            c.drawString(col_width + 30, segment_y, f"{sub_key}: {sub_value}")
            segment_y -= 20
    # --- Footer note ---
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(gray)
    c.drawCentredString(
        width / 2,
        10,
        "This report was automatically generated by a Deep Learning model "
        "using patient information and CMRI images."
    )
    print("Save report to: ", pdf_file)
    c.save()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_info_path", type=str, default="ACDC/database/testing/patient101/Info.cfg")
    parser.add_argument("--patient_seg_path", type=str, default="ACDC/database/testing/patient101/")
    args = parser.parse_args()
    create_acdc_report("patient_acdc_report.pdf", args.patient_info_path, args.patient_seg_path)
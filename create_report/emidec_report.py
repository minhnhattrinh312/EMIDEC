# import canvas
from inspect import ArgInfo
from reportlab.pdfgen import canvas
# import blue color
from reportlab.lib.colors import *
from reportlab.lib.utils import ImageReader
import io
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("./")
from segment2d import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_emidec = torch.load("tiramisu_emidec.pt", weights_only=False)
model_emidec.eval()
model_emidec = model_emidec.to(device)

def read_patient_EMIDEC(file_path):
    patient_data = {}
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            # if it is the first line 'Case N006' parts = ['Case', 'N006']
            if line.startswith('Case'):
                parts = line.strip().split(' ')
                patient_data['Case'] = parts[1]
            else:
                # Remove the � character and split the line
                clean_line = line.replace('�', '').strip()
                parts = clean_line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    # Use the key as the dictionary key, converting to a standard format
                    patient_data[key] = value
    return patient_data


def create_emidec_report(pdf_file, clinical_data_path, segmentation_data_path=None, model_emidec=model_emidec):
    print("Creating report for patient with clinical data path: ", clinical_data_path)
    print("Creating report for patient with segmentation data path: ", segmentation_data_path)

    # --- Load data ---
    patient_dict = read_patient_EMIDEC(clinical_data_path)
    image, _, header = load_nii(segmentation_data_path)
    # image, restore_info = crop_resize_image(image, 256)
    data = preprocess_data_nii(segmentation_data_path)
    print("Predicting segmentation data...")
    # calculate the time of predicting segmentation data
    start_time = time.time()
    seg = predict_data_model_emidec(data, model_emidec, min_size_remove=500).astype(np.uint8)
    end_time = time.time()
    print(f"Time taken to predict segmentation data: {end_time - start_time} seconds")
    # seg = crop_resize_mask(seg, restore_info)
    print("Segmentation data predicted successfully")
    w, h = image.shape[:2]
    image = image[w//2-64:w//2+64, h//2-64:h//2+64, :]
    seg = seg[w//2-64:w//2+64, h//2-64:h//2+64, :]

    cavity = (seg == 1)
    myocardium = (seg == 2) + (seg == 3) + (seg == 4)
    infarction = (seg == 3) + (seg == 4)
    no_reflow = (seg == 4)
    cavity_volume = round(make_volume(cavity, header.get_zooms()) / 1000, 2)
    myocardium_volume = round(make_volume(myocardium, header.get_zooms()) / 1000, 2)
    infarction_volume = round(make_volume(infarction, header.get_zooms()) / 1000, 2)
    no_reflow_volume = round(make_volume(no_reflow, header.get_zooms()) / 1000, 2)
    # compute the volume ratio
    infarction_ratio = round(infarction_volume / myocardium_volume, 2)
    no_reflow_ratio = round(no_reflow_volume / myocardium_volume, 2)
    # --- Choose middle slices ---
    z_slices = [0,seg.shape[-1]//2,-1]

    # --- Create PDF canvas ---
    pagesize = (600, 600)
    c = canvas.Canvas(pdf_file, pagesize=pagesize)
    width, height = pagesize


    # --- Header ---
    c.drawImage('figures/logo1.png', 10, height - 90, width=90, height=90)

    c.setFont("Helvetica-Bold", 30)
    # make it center horizontally
    c.drawCentredString(width / 2, height - 50, "PATIENT REPORT")

    # --- Patient info ---
    infor_list = ["Case", "Age", "Sex"]
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
        ax[i].imshow(image[..., z], cmap="gray", alpha=0.7)
        ax[i].imshow(seg[..., z], cmap="jet", alpha=0.5, vmin=0, vmax=5)
        ax[i].axis("off")
        # adjust the space between the images   
    plt.subplots_adjust(wspace=0.05)

    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # --- Draw combined image on PDF ---
    image_x = 50
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
    c.setFillColor(blue)
    c.rect(legend_x, legend_y, 10, 10, fill=1, stroke=0)
    c.setFillColor(HexColor("#5ec8af"))
    c.rect(legend_x, legend_y - 20, 10, 10, fill=1, stroke=0)
    c.setFillColor(HexColor("#bdd76c"))
    c.rect(legend_x, legend_y - 40, 10, 10, fill=1, stroke=0)
    c.setFillColor(HexColor("#ce9550"))
    c.rect(legend_x, legend_y - 60, 10, 10, fill=1, stroke=0)
    c.setFillColor(black)
    c.drawString(legend_x + 15, legend_y + 1, "Cavity")
    c.drawString(legend_x + 15, legend_y - 19, "Myocardium")
    c.drawString(legend_x + 15, legend_y - 39, "Infarction")
    c.drawString(legend_x + 15, legend_y - 59, "No Reflow")

    # --- Clinical data section ---
    col_width = width / 2
    clinical_y = height - 340
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
    segment_y = height - 340
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(red)
    c.drawCentredString(col_width + (col_width / 2), segment_y, "Segmentation Data")
    segment_y -= 20
    c.setFont("Helvetica", 12)
    segment_data = {
        "Volume": {
            "Cavity": f"{cavity_volume} mL",
            "Myocardium": f"{myocardium_volume} mL",
            "Infarction": f"{infarction_volume} mL",
            "No Reflow": f"{no_reflow_volume} mL",
            "Fraction of myocardium infarcted": f"{infarction_ratio}%",
            "Fraction of myocardium with no-reflow": f"{no_reflow_ratio}%"
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
    c.save()
    print("Save report to: ", pdf_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_info_path", type=str, default="emidec-dataset-1.0.1/Case P045.txt")
    parser.add_argument("--patient_seg_path", type=str, default="emidec-dataset-1.0.1/Case_P045/Images/Case_P045.nii.gz")
    args = parser.parse_args()
    create_emidec_report("patient_emidec_report.pdf", args.patient_info_path, args.patient_seg_path)
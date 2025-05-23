import glob
from torch.utils.data import Dataset
import torch
import numpy as np
from segment2d.utils import *
from segment2d.config import cfg
import nibabel as nib


class Image_Loader(Dataset):

    def __init__(self, train_path="", list_subject=[]):
        if list_subject:
            self.listName = list_subject
        else:
            self.listName = glob.glob(train_path)

    def __len__(self):
        return len(self.listName)

    def __getitem__(self, idx):
        data = np.load(self.listName[idx])
        image, mask = data["image"], data["mask"]
        # convert image, mask to tensor
        image = torch.tensor(image.transpose(-1, 0, 1), dtype=torch.float32)

        mask = torch.from_numpy(mask[None])
        return image, mask.float()


class Test_Volume_Loader(Dataset):

    def __init__(self, list_subject=[]):

        self.listName = list_subject

    def __len__(self):
        return len(self.listName)

    def __getitem__(self, idx):
        image_path = self.listName[idx]
        data = dict()


        image = min_max_normalize(image)
        if "MnM" in image_path:
            image = nib.load(image_path).get_fdata()
            mask = nib.load(image_path.replace(".nii.gz", "_gt.nii.gz")).get_fdata()

        elif "emidec-dataset" in image_path:
            image = nib.load(image_path).get_fdata()
            mask = nib.load(image_path.replace("Images", "Contours")).get_fdata()

        
        if "MnM" in image_path:
            padded_image, crop_index, padded_index = pad_background(image, dim2pad=cfg.DATA.DIM2PAD, x_shift=-50)
        elif "emidec-dataset" in image_path:
            padded_image, crop_index, padded_index = pad_background(image, dim2pad=cfg.DATA.DIM2PAD)
            
        data["crop_index"] = crop_index
        data["padded_index"] = padded_index
        data["mask"] = mask.astype(np.int64)
        data["original_shape"] = image.shape
        batch_images = []
        for i in range(padded_image.shape[-1]):
            slice_inputs = padded_image[..., i : i + 1]  # shape (224, 224, 1)
            slices_image = torch.from_numpy(slice_inputs.transpose(-1, 0, 1))  # shape (1, 224, 224)
            batch_images.append(slices_image)

        batch_images = torch.stack(batch_images).float()  # shape (9,1, 224, 224)
        data["image"] = batch_images
        return data

import numpy as np
from skimage.morphology import remove_small_objects


# def min_max_normalize(volume):
#     volume = (volume - np.min(volume)) / (np.max(volume)-np.min(volume)) * 255.0
#     return volume.astype(np.uint8)
def min_max_normalize(image, low_perc=0.05, high_perc=99.05):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    """
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    return image


def pad_background(image, dim2pad=(128, 128), x_shift=0, y_shift=0):
    """
    Pads the image to dim2pad, cropping non-zero regions and shifting the crop by x_shift (width) and y_shift (height) pixels from the center.
    To invert the operation, use:
    inverted_image = np.zeros_like(image)
    inverted_image[crop_index] = padded_image[padded_index]
    """
    if len(image.shape) == 3:
        dim2pad = (dim2pad[0], dim2pad[1], image.shape[2])
    elif len(image.shape) == 4:
        dim2pad = (dim2pad[0], dim2pad[1], image.shape[2], image.shape[3])

    # Use np.nonzero to find the indices of all non-zero elements in the image
    nz = np.nonzero(image)

    # Get the minimum and maximum indices along each axis
    min_indices = np.min(nz, axis=1)
    max_indices = np.max(nz, axis=1)

    # Crop the image to only include non-zero values
    crop_index = tuple(slice(imin, imax + 1) for imin, imax in zip(min_indices, max_indices))
    cropped_img = image[crop_index]
    padded_image = np.zeros(dim2pad)
    # Crop further if any axis is larger than dim2pad
    crop_index_new = crop_index
    if cropped_img.shape[0] > dim2pad[0]:
        cx, cx_pad = cropped_img.shape[0] // 2, dim2pad[0] // 2
        # Shift the crop by y_shift pixels from the center
        cx_shifted = cx + x_shift
        start_x = max(0, cx_shifted - cx_pad)  # Ensure start index is non-negative
        end_x = min(cropped_img.shape[0], start_x + dim2pad[0])  # Ensure end index doesn't exceed image height
        cropped_img = cropped_img[start_x:end_x, :, :]
        crop_index_new = (
            slice(crop_index[0].start + start_x, crop_index[0].start + end_x),
            crop_index[1],
            crop_index[2],
        )

    if cropped_img.shape[1] > dim2pad[1]:
        cy, cy_pad = cropped_img.shape[1] // 2, dim2pad[1] // 2
        # Shift the crop by x_shift pixels from the center
        cy_shifted = cy + y_shift
        start_y = max(0, cy_shifted - cy_pad)  # Ensure start index is non-negative
        end_y = min(cropped_img.shape[1], start_y + dim2pad[1])  # Ensure end index doesn't exceed image width
        cropped_img = cropped_img[:, start_y:end_y, :]
        crop_index_new = (
            crop_index_new[0],
            slice(crop_index[1].start + start_y, crop_index[1].start + end_y),
            crop_index_new[2],
        )

    # Calculate the amount of padding needed along each axis
    pad_widths = [(padded_image.shape[i] - cropped_img.shape[i]) // 2 for i in range(len(cropped_img.shape))]
    # Pad the image with zeros
    padded_index = tuple(
        slice(pad_widths[i], pad_widths[i] + cropped_img.shape[i]) for i in range(len(cropped_img.shape))
    )

    padded_image[padded_index] = cropped_img

    return padded_image, crop_index_new, padded_index


def pad_background_with_index(image, crop_index_new, padded_index, dim2pad=(128, 128)):
    if len(image.shape) == 3:
        dim2pad = (dim2pad[0], dim2pad[1], image.shape[2])
    elif len(image.shape) == 4:
        dim2pad = (dim2pad[0], dim2pad[1], image.shape[2], image.shape[3])
    padded_image = np.zeros(dim2pad)
    crop_image = image[crop_index_new]
    padded_image[padded_index] = crop_image
    return padded_image


def invert_padding(original_shape, padded_image, crop_index, padded_index):
    # crop the padded image to the size of the original image
    cropped_img = padded_image[padded_index]

    # create an array of zeros with the same shape as the original image
    inverted_image = np.zeros(original_shape)

    # insert the cropped padded image into the center of the array of zeros
    inverted_image[crop_index] = cropped_img

    return inverted_image


def remove_small_elements(segmentation_mask, min_size_remove=3):
    # Convert segmentation mask values greater than 0 to 1
    pred_mask = segmentation_mask > 0
    # Remove small objects (connected components) from the binary image
    mask = remove_small_objects(pred_mask, min_size=min_size_remove)
    # Multiply original segmentation mask with the mask to remove small objects
    clean_segmentation = segmentation_mask * mask
    return clean_segmentation

import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

def get_border(image: np.array, idx_k: int, g_threshold: float, is_top: bool) -> int:
    """Returns the border between the background and the cable.

    This method finds the largest rectange that is completly inside the cable.
    Since the background is green, given a window, the value of green must be higher than
    the value of red and blue. With a threshold, we can decide if the window is mainly green.
    Since the cable is angled sometimes, we check for a window containing only the cable on right and
    left side of the image and find the lowest index for top and highest index for bottom to crop only the cable.

    Args:
        image (np.array): The image to be cropped.
        idx_k (int): The size of the window to inspect the green colour intensity.
        g_threshold (int): Threshold to decide whether the window is green.
        is_top (bool): Flag to indicate whether to find the top border or the bottom border.

    Returns:
        border_index (int): The index of the border.
    """
    index_range = range(0, image.shape[0] // 2) if is_top else range(image.shape[0] // 2, 0)
    border_index = 0 if is_top else image.shape[0]
    
    if is_top:
        for i in index_range:
            right_mean_r, right_mean_g, right_mean_b = image[i + 1][-idx_k:].mean(axis=0)
            # print(image[i + 1][-idx_k:].mean(axis=0))
            if right_mean_g < g_threshold * (right_mean_r + right_mean_b):
                return i
    else:    
        i = image.shape[0]
        while i > image.shape[0] / 2:
            left_mean_r, left_mean_g, left_mean_b = image[i - 1][:idx_k].mean(axis=0)
            if left_mean_g < g_threshold * (left_mean_r + left_mean_b):
                return i
            i -= 1
    
    return border_index
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = os.path.join(os.getcwd(), "CableInspect-AD")

cableinspect_dataframe = pd.read_csv(root / Path("labels.csv"))

image_paths = cableinspect_dataframe.image_path # Path to the image
label_indices = cableinspect_dataframe.label_index # Whether image is normal (0) or anomalous (1) 
cable_ids = cableinspect_dataframe.cable_id # Either C01, C02 or C03
mask_paths = cableinspect_dataframe.mask_path # Path to the Mask, I only have the paths for C01
pass_ids = cableinspect_dataframe.pass_id # Which set of cable pass it is (Either 1, 2 or 3)


total = 0
training_images_one = [] # For the Pass Ids
training_images_two = []
training_images_tri = []
anomaly_images = []
gt = []
dic_t = {"1": 0, "2": 0, "3": 0}
dic_a = {"1": 0, "2": 0, "3": 0}


for image_path, cable_id, anomaly_label, mask_path, pass_id in zip(image_paths, cable_ids, label_indices, mask_paths, pass_ids):
    
    image = cv2.imread(os.path.join(root, image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 1080, 1920, 3
    
    # We remove most of the unecesary background being green
    top_index = get_border(image, 5, 0.6, True)
    bottom_index = get_border(image, 5, 0.55, False)
    print(f"top_index: {top_index} bottom_index: {bottom_index}")

    leftover = 512 - (bottom_index - top_index)
    leftover = leftover // 2
    image = image[top_index - leftover:bottom_index + leftover, :, :] # [512, 1920, 3]
    
    image = cv2.resize(image, dsize=(768, 384), interpolation=cv2.INTER_AREA) # width, height

    tensor_rgb = torch.from_numpy(image) / 127.5 - 1 # Put directly in [-1, 1] for LPIPS later
    
    """Display Image if necessary."""
    # if anomaly_label == 1 and cable_id == 'C01':
    #     plt.imshow((tensor_rgb + 1)/ 2)
    #     plt.show()
    
    tensor_rgb = tensor_rgb.permute((2, 0, 1))
    tensor_rgb = tensor_rgb.unsqueeze(0)

    total += 1
    if anomaly_label == 0:
        if pass_id == 1:
            training_images_one.extend(tensor_rgb)
        elif pass_id == 2:
            training_images_two.extend(tensor_rgb)
        else:
            training_images_tri.extend(tensor_rgb)
        dic_t[str(pass_id)] += 1
    else:
        dic_a[str(pass_id)] += 1

        if pass_id == 1: # I only have the paths for masks in pass_id 1 for some reason?
            anomaly_images.extend(tensor_rgb)
            
            # We need the mask as well
            mask = cv2.imread(os.path.join(root, mask_path), cv2.IMREAD_GRAYSCALE)
            mask = mask[top_index - leftover:bottom_index + leftover, :] # Crop to [512, 1920, 3], as background is all green, like the regular image
            mask = cv2.resize(mask, dsize=(768, 384), interpolation=cv2.INTER_AREA) # width, height
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(-1)
            mask = mask.permute((2, 0, 1))
            mask = mask.unsqueeze(0)

            mask = torch.where(mask > 0, 1, 0)
            gt.extend(mask)
            
# print(total) # 6023
# print(dic_t) # {'1': 603, '2': 613, '3': 943}
# print(dic_a) # {'1': 976, '2': 1077, '3': 1811}

normal_one = torch.stack(training_images_one, dim=0) # [603, 3, 384, 768]
normal_two = torch.stack(training_images_two, dim=0) # [613, 3, 384, 768]
normal_tri = torch.stack(training_images_tri, dim=0) # [943, 3, 384, 768]
anomalies = torch.stack(anomaly_images, dim=0) # [976, 3, 384, 768]
a_gt = torch.stack(gt, dim=0) # [976, 1, 384, 768]

normal = torch.cat([normal_one, normal_two, normal_tri], dim=0) # [2159, 3, 384, 768]

torch.save(normal, os.path.join(f"allcables01_train.pt"))
torch.save(anomalies, os.path.join(f"allcables01_test.pt"))
torch.save(a_gt, os.path.join(f"allcables01_test_gt.pt"))
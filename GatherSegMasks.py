import os
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

def show_img_mask(img, mask):
    # _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
    
    # axes[0].imshow(img)
    # axes[0].set_title('Image')
    
    # axes[1].imshow(mask)
    # axes[1].set_title('Mask')
    
    plt.imshow(mask)
    plt.show()

def acquire_segAD_masks(image, show_mask = False):
    image = image.cpu().numpy()
    masks = mask_generator.generate(image)
    
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    if show_mask:
        show_img_mask(image, sorted_masks[0]['segmentation'])
    
    # First index mask is a good one, since the object is one colour, the background is another
    return sorted_masks[0]['segmentation'].astype(int)

def loop_thru_data(dataset, show_mask = False):
    N = len(dataset) * dataset.batch_size
    batch = next(iter(dataset))
    _, _, H, W = batch.shape
    
    segmentation_masks = np.empty((N, 1, H, W), dtype=np.uint8)
    
    for i, image in enumerate(dataset):
        image = image.to(device)
        mask = acquire_segAD_masks(image[0].permute((1, 2, 0)), show_mask)
        mask = mask[np.newaxis, ...]
        segmentation_masks[i] = mask
    
    return torch.from_numpy(segmentation_masks)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Segment Anything Model (SAM)
sam = sam_model_registry["vit_h"](checkpoint=os.path.join(os.getcwd(), "sam_vit_h_4b8939.pth")).to(device)
mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=400)

# Load Data to acquire each individual segment mask (in this case binary)
'''Cables Dataset.'''
normal = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allcables01_train.pt'), weights_only=True).to(torch.float32) # [2159, 3, 384, 768]
anomalies = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allcables01_test.pt'), weights_only=True).to(torch.float32) # [976, 3, 384, 768]

'''MVTec3D-AD dataset.'''
# normal = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allobjects_train.pt'), weights_only=True).to(torch.float32) # [2656, 3*224*224 * 2]
# anomalies = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allobjects_test.pt'), weights_only=True).to(torch.float32) # [1197, 3*224*224 * 2]

# normal = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_train.pt'), weights_only=True).to(torch.float32) # [2656, 6, 224, 224]
# anomalies = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_test.pt'), weights_only=True).to(torch.float32) # [1197, 6, 224, 224]

# normal = normal[:, :3, :, :] # Just need the rgb portion for the mask
# anomalies = anomalies[:, :3, :, :]
''''''

batch_size = 1 # SAM seems to not do batching, and I only really need to run this file once per dataset, so not bothering changing it right now
normal = DataLoader(normal, batch_size=batch_size, shuffle=False, pin_memory=False)
anomalies = DataLoader(anomalies, batch_size=batch_size, shuffle=False, pin_memory=False)

show_mask = True

masks = loop_thru_data(normal, show_mask)
torch.save(masks, os.path.join(f"segmasks_normal_mvtec.pt"))

print("Done Normal!")

masks = loop_thru_data(anomalies, show_mask)
torch.save(masks, os.path.join(f"segmasks_anomalous_mvtec.pt"))

print('Done Anomalies!')


from PIL import Image
import os
import torchvision.transforms as transforms
import torch
from tifffile import imread as timread

def acquireImages(dataset: str, objects: list, data_samples: int):
    """Summary: Find images in directory and amalgamate them all into numpy arrays and a list.
    There are different types of images. [bagel, cable_gland, carrot, cookie, dowel, foam, peach, potato, rope, tire]
    Each image has xyz coordinates and rgb values, in seperate folders.
    
    This file/function runs in the same directory as "mvtec_3d_anomaly_detection" folder. 
    
    - mvtec_3d_anomaly_detection
    - Generate3DDatasets.py
    
    => Prepares RGB, Point Clouds & Ground Truth Images in tensors and stored them on disk. I place them in a folder named "ObjectTensors".

    Args:
        dataset (str): The type of dataset the model will train on. Either train, test or validation
        objects (list): The different objects the mvtec 3d anomaly detection dataset provides and 
        which you want the VQGAN-LEDGM to train/test on.
        data_samples (int): The number of samples in the dataset, 2656 for train, 294 for validation, 1197 for test
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    rgb = torch.ones((data_samples, 3, 224, 224))
    rgb_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    
    xyz = torch.ones((data_samples, 3, 224, 224))
    
    gt_transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
                                                   transforms.ToTensor()])
    
    # 1197 (Only test folder has anomlies)
    gt = torch.ones((1197, 1, 224, 224))
    
    labels = torch.ones((data_samples, 1))
    
    # Index counters to store into the pre-initialized tensors
    labels_index = 0 # Used for Image Wise classification labels 0-9 for 10 classes
    rgb_index = 0
    xyz_index = 0
    gt_index = 0
    for obj in objects:
            for curpath, _, files in os.walk(os.path.join(os.getcwd(), f"mvtec_3d_anomaly_detection\\{obj}\\{dataset}")):
                files = [fi for fi in files if fi.endswith((".png", ".tiff"))]
                for fi in files:
                    if curpath.endswith("gt"):
                        image = Image.open(f"{curpath}\\{fi}")
                        gt_tensor = gt_transform(image) # [1, 224, 224]
                        
                        gt_tensor = torch.where(gt_tensor > 0, 1, 0) # [1, 224, 224]
                        
                        gt[gt_index] = gt_tensor
                        gt_index += 1
                        
                    elif curpath.endswith("rgb"):
                        image = Image.open(f"{curpath}\\{fi}")
                        rgb_tensor = rgb_transform(image) # [3, 224, 224]
                        
                        rgb[rgb_index] = rgb_tensor
                        
                        labels[labels_index] = objects.index(obj)
                        
                        rgb_index += 1
                        labels_index += 1
                    else:
                        xyz_tensor = torch.tensor(timread(f"{curpath}\\{fi}"))   # [H, W, 3]                 
                        
                        torch_organised_pc = xyz_tensor.permute(2, 0, 1).unsqueeze(dim=0) # Unsqueeze to add a batch size dimension  # (224, 224, 3)                       
                        
                        torch_resized_organised_pc = torch.nn.functional.interpolate(torch_organised_pc, size=(224, 224), mode='nearest')
                        torch_resized_organised_pc = torch_resized_organised_pc.squeeze(dim=0).permute(1, 2, 0)
                        
                        datamin = torch.amin(torch_resized_organised_pc, dim=-1, keepdim=True)
                        datamax = torch.amax(torch_resized_organised_pc, dim=-1, keepdim=True)
                        
                        torch_resized_organised_pc = (torch_resized_organised_pc - datamin) / (datamax - datamin)
                    
                        torch_resized_organised_pc = torch.nan_to_num(torch_resized_organised_pc, nan=0)
                        
                        xyz[xyz_index] = torch_resized_organised_pc.permute(2, 0, 1)
                            
                        xyz_index += 1
    
    torch.save(torch.cat((rgb, xyz), dim=1), f'ObjectTensors\\MVTEC3D_{dataset}.pt')
    torch.save(labels, f'ObjectTensors\\MVTEC3D_{dataset}_labels.pt')
    if dataset == 'test':
        torch.save(gt, f'ObjectTensors\\MVTEC3D_test_gt.pt')

# Tensors of these objects are different sizes, so we need preprocessing to bring them to a consistent shape
objects = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"]

acquireImages("train", objects, 2656)
acquireImages("test", objects, 1197)
acquireImages("validation", objects, 294)
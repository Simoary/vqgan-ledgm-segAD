import torch
import torch.nn as nn
from VQGAN_LEDGM import VQGAN_LEDGM
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from AutoT import AutoregressiveTransformer
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import yaml
from torchvision import transforms
from pro_curve_util import compute_pro
from generic_util import trapezoid
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""Prepare Datasets.

Training added to Testing. Some data in the train dataset are added to the test dataset, since the test dataset from disk is entirely anomalies (for CableInspectAD).
Ground Truth contains the masks to where the anomalies are locaated in the original rgb images.
"""
"""CableInspect-AD"""
# Training and test datasets begin in range [-1, 1] for the LPIPS. Model input trained with [-1, 1].
# train = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allcables01_train.pt'), weights_only=True).to(torch.float32) # [num_images, 3, H, W]
# test = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allcables01_test.pt'), weights_only=True).to(torch.float32)
# gt = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allcables01_test_gt.pt'), weights_only=True).to(torch.float32)

"""MVTEC3D-AD"""
labels = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_train_labels.pt'), weights_only=True) # [2656, 1]
train = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_train.pt'), weights_only=True).to(torch.float32) # [2656, 6, 224, 224]

test = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_test.pt'), weights_only=True).to(torch.float32) # [1197, 6, 224, 224]
gt = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_test_gt.pt'), weights_only=True).to(torch.float32) # [1197, 1, 224, 224]
gt = gt.reshape(-1, 224, 224)

# rescale between [-1 and 1]
train = 2 * train - 1
test = 2 * test - 1

# Scale to RGB
# pre_train = (pre_train + 1) / 2

# # Rescale images between -1 and 1
# pre_train = 2 * pre_train - 1
# test = 2 * test - 1
"""End Prepare Datasets."""

compression_factor = 8
LATENT_SPACE_HEIGHT = int(train.shape[2] / compression_factor)
LATENT_SPACE_WIDTH = int(train.shape[3] / compression_factor)

residual_train = torch.zeros((train.shape[0], 1, train.shape[2], train.shape[3]))
residual_test = torch.zeros((test.shape[0], 1, test.shape[2], test.shape[3]))

"""Prepare Anomaly Detection Models. VQGAN-LEDGM & Autoregressive Transformer, which work together to perform anomaly detection (AD).
The VQGAN-LEDGM is trained on normal images, producing a quantised latent space for normal images. The autoregressive transformer
learns the prior distribution of these normal images, the sequences which exist in the normal latent space. The idea of this 2 model
team is that anomalous images will deviate from the normal latent sequences, and the autoregressive transformers will find those anomalous 
patterns in the sequences produced from the VQGAN-LEDGM and resample them for a normal sequence. We can then compare the two images for AD."""
"""1. Parameters for the VQGAN-LEDGM Hybrid."""

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
vql_params = config['vqganledgm']

savedir = f"28latentspace_{vql_params['codebook_size']}_5degrees_rgbxyz" #f"cables_recon_{vql_params['codebook_size']}"

vqgan_info = f"{vql_params['emb_dim']}_{vql_params['z_channels']}_{vql_params['codebook_size']}_{vql_params['beta_vq']}_{vql_params['discriminator_threshold']}"

vql = VQGAN_LEDGM(*vql_params.values(), device=device, double=True)
vql.load_pretrained_weights(os.path.join(os.getcwd(), "Model", f"VQGAN_{savedir}_epoch1500_codedim{vql_params['emb_dim']}_zchan{vql_params['z_channels']}_csize{vql_params['codebook_size']}_disc_threshold{vql_params['discriminator_threshold']}.pt"))
vql.float().to(device)
vql.eval()


"""2. Parameters for the Autoregresssive Transformer."""
autot_params = config['transformer']
max_seq_len = LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH + 1

autot_details = f"{autot_params['num_blocks']}blocks_{autot_params['n_heads']}attnheads_{autot_params['d_model']}d_model"

autot = AutoregressiveTransformer(autot_params['num_blocks'], vql_params['codebook_size'], max_seq_len, autot_params['n_heads'], autot_params['d_model'], device=device)
checkpoint = torch.load(os.path.join(os.getcwd(), "AutoTWeights", f"autot_{savedir}_autot{autot_details}_vqgan{vqgan_info}_150epochs.pt"), weights_only=True, map_location='cuda:0')
autot.load_state_dict(checkpoint['autot_state_dict'])
autot.eval()

autot_reverse = AutoregressiveTransformer(autot_params['num_blocks'], vql_params['codebook_size'], max_seq_len, autot_params['n_heads'], autot_params['d_model'], device=device)
checkpoint = torch.load(os.path.join(os.getcwd(), "AutoTWeights", f"Reverseautot_{savedir}_autot{autot_details}_vqgan{vqgan_info}_150epochs.pt"), weights_only=True, map_location='cuda:0')
autot_reverse.load_state_dict(checkpoint['autot_state_dict'])
autot_reverse.eval()
"""End Loading Transformers."""

batch_size = 1
train = DataLoader(train, batch_size=batch_size, shuffle=False, pin_memory=False)
test = DataLoader(TensorDataset(test, gt), batch_size=batch_size, shuffle=False, pin_memory=False)
# test = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=False)

cross_entropy = nn.CrossEntropyLoss(reduction='none')
blur_transform = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
BOS_TOKEN = torch.full((batch_size, 1), vql_params['codebook_size'], device=device)

print("All Systems Go!")

"""Anomaly Detection Code."""

def to_rgb(image):
    return (image + 1) / 2

def display_images(rgb, rgb_recon, rgb_residual, xyz, xyz_recon, xyz_residual, anomaly_map, ground_truth):
    
    _, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 9))
    
    axes[0][0].imshow(rgb[0].cpu().permute((1, 2, 0)))
    axes[0][0].set_title('Original X')
    axes[0][0].axis('off')
    
    axes[0][1].imshow(rgb_recon[0].cpu().permute((1, 2, 0))) # Here
    axes[0][1].set_title('healthy X')
    axes[0][1].axis('off')
    
    axes[0][2].imshow(rgb_residual[0].cpu().permute((1, 2, 0)))
    axes[0][2].set_title('Residual')
    axes[0][2].axis('off')
    
    axes[0][3].imshow(anomaly_map[0].cpu().permute((1, 2, 0)))
    axes[0][3].set_title('Anomaly Map')
    axes[0][3].axis('off')
    
    axes[1][0].imshow(xyz[0].cpu().permute((1, 2, 0)))
    axes[1][0].set_title('Original X')
    axes[1][0].axis('off')
    
    axes[1][1].imshow(xyz_recon[0].cpu().permute((1, 2, 0))) # Here
    axes[1][1].set_title('healthy X')
    axes[1][1].axis('off')
    
    axes[1][2].imshow(xyz_residual[0].cpu().permute((1, 2, 0)))
    axes[1][2].set_title('Residual')
    axes[1][2].axis('off')
    
    axes[1][3].imshow(ground_truth.cpu().permute((1, 2, 0)))
    axes[1][3].set_title('Ground Truth')
    axes[1][3].axis('off')
    
    plt.tight_layout() # better spacing
    plt.show()
    
def cableDisplay(rgb, healthy, anomaly_map, gt):
    
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 9))
    
    axes[0].imshow(rgb[0].cpu().permute((1, 2, 0)))
    axes[0].set_title('Original X')
    axes[0].axis('off')

    axes[1].imshow(healthy[0].cpu().permute((1, 2, 0)))
    axes[1].set_title('healthy X')
    axes[1].axis('off')
    
    axes[2].imshow(anomaly_map[0].cpu().permute((1, 2, 0)))
    axes[2].set_title('Anomaly Map')
    axes[2].axis('off')
    
    axes[3].imshow(gt[0].cpu().permute((1, 2, 0)))
    axes[3].set_title('Ground Truth')
    axes[3].axis('off')
    
    plt.tight_layout() # better spacing
    plt.show()
    

def hippocratic_oath(indices, transformer, reverse: bool = False, p_threshold: float = 0.97):
    
    nll_threshold = - torch.log(torch.Tensor([p_threshold])).to(device)
    
    if reverse:
        indices = torch.flip(indices, dims=[-1])
    
    input_seq = torch.cat([BOS_TOKEN[:indices.shape[0], :], indices], dim=-1) # [batch, LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH + 1]
    
    choices, _ = transformer(input_seq) # [batch, LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH + 1, codebook_size]
    
    nll_per_index = cross_entropy(choices[:, 1:, :].reshape((-1, vql_params['codebook_size'])), indices.reshape(-1)) # NLL per index
    nll_per_index = nll_per_index.reshape(choices.shape[0], -1) # [batch, LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH]    
    # nll_per_image = nll_per_index.mean(dim=-1) # [batch, 1] Could be used as an image level anonmaly score
    
    resample_mask = (nll_per_index < nll_threshold).int()   
    
    healthy_sequence = resample(transformer, resample_mask[0], indices[0])
    
    if reverse:
        healthy_sequence = torch.flip(healthy_sequence, dims=[-1])
    
    healthy_sequence = healthy_sequence.reshape(batch_size, LATENT_SPACE_HEIGHT, LATENT_SPACE_WIDTH)
    z_q = vql.indices_to_z_q(healthy_sequence)
    healthy_recon = vql.decode(z_q)
    
    return healthy_recon, resample_mask.reshape(1, 1, LATENT_SPACE_HEIGHT, LATENT_SPACE_WIDTH)

def resample(transformer, resample_mask, min_indices, temperature: float = 1):
    """Resample operation to create healthy sequences."""
    cache = None
    
    indices_to_resample = torch.where(resample_mask == 0)[0]
    
    healthy_sequence = min_indices.clone().unsqueeze(0)
    healthy_sequence = torch.cat((BOS_TOKEN[:batch_size, :], healthy_sequence), dim=-1)
    
    for i in indices_to_resample:
        prefix = healthy_sequence[:, :i + 1]
        logits, _ = transformer(prefix, cache)
        
        next_logits = logits[:, -1, :] # [batch, 512]
        probs = torch.softmax(next_logits / temperature, dim=-1)
        codevector = torch.multinomial(probs, num_samples=1).squeeze(-1)
        healthy_sequence[:, i + 1] = codevector
    
    return healthy_sequence[:, 1:]

def compute_anomaly_maps(dataloader, residual_data, p = 0.97):

    amaps = []
    gtm = []
    
    for i, (x, gt) in enumerate(dataloader):
        with torch.no_grad():
            _, _, min_encoding_indices = vql.encode(x.to(device), 0) # [batch * (LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH)]
            
            '''Reconstruction code for anomaly maps.'''
            min_encoding_indices = torch.reshape(min_encoding_indices, (-1, LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH))

            # Healthy Sequence
            healthy_recon, _ = hippocratic_oath(min_encoding_indices, autot, p_threshold=p)
            
            # Reverse Healthy Sequence
            reverse_healthy_recon, _ = hippocratic_oath(min_encoding_indices, autot_reverse, reverse=True, p_threshold=p)
            
            x, healthy_recon, reverse_healthy_recon = to_rgb(x), to_rgb(healthy_recon), to_rgb(reverse_healthy_recon)
            
            '''MVTec3D-AD'''
            x_rgb = x[:, :3, :, :]
            x_xyz = x[:, 3:, :, :]
            h_rgb = healthy_recon[:, :3, :, :]
            h_xyz = healthy_recon[:, 3:, :, :]
            
            r_h_rgb = reverse_healthy_recon[:, :3, :, :]
            r_h_xyz = reverse_healthy_recon[:, 3:, :, :]
            
            rgb_residual = torch.abs(x_rgb - h_rgb.cpu()).mean(dim=1, keepdim=True)
            xyz_residual = torch.abs(x_xyz - h_xyz.cpu()).mean(dim=1, keepdim=True)
            
            r_rgb_residual = torch.abs(x_rgb - r_h_rgb.cpu()).mean(dim=1, keepdim=True)
            r_xyz_residual = torch.abs(x_xyz - r_h_xyz.cpu()).mean(dim=1, keepdim=True)
            
            # For AUPRO (Just this line)
            residuals = rgb_residual + xyz_residual + r_rgb_residual + r_xyz_residual # / 4

            threshold = 0.17
            anomaly_map_rgb = (rgb_residual > threshold).float()
            anomaly_map_xyz = (xyz_residual > threshold).float()
            
            r_anomaly_map_rgb = (r_rgb_residual > threshold).float()
            r_anomaly_map_xyz = (r_xyz_residual > threshold).float()
            
            
            anomaly_map = (anomaly_map_rgb + anomaly_map_xyz + r_anomaly_map_rgb + r_anomaly_map_xyz) # / 4
            anomaly_map = blur_transform(anomaly_map)
            
            # display_images(x_rgb, h_rgb, anomaly_map_rgb, x_xyz, h_xyz, anomaly_map_xyz, anomaly_map, gt)
            '''End MVTec3D-AD.'''
            
            '''Cable Dataset'''
            # healthy_residual = torch.abs(x - healthy_recon.cpu()).mean(dim=1, keepdim=True)
            # reverse_healthy_residual = torch.abs(x - reverse_healthy_recon.cpu()).mean(dim=1, keepdim=True)
            
            # residuals = healthy_residual + reverse_healthy_residual 
            
            # threshold = 0.125
            # anomaly_map = (healthy_residual > threshold).float() + (reverse_healthy_residual > threshold).float()
            # anomaly_map = blur_transform(anomaly_map)
            
            # cableDisplay(x, healthy_recon, anomaly_map, gt)
            '''End Cable.'''
            
            # Store anomaly_map
            residual_data[i] = anomaly_map
            
            '''For Calculating AUPRO'''
            a_map = residuals.mean(dim=1)
            amaps.append(a_map[0].numpy())
            
            gt_map = gt.squeeze(1)
            gtm.append(gt_map[0].numpy())
            
    return residual_data, amaps, gtm

# Probability for Resampling: 0.97 for cables, 0.99 for 3D
p = 0.99 
    
# Get Residuals For SegAD
# residuals, _, _ = compute_anomaly_maps(train, residual_train, p)
# torch.save(residuals, os.path.join(f"residual_MVTEC_train.pt"))
# print("Done Training!")

residuals, amaps, gtm = compute_anomaly_maps(test, residual_test, p)

'''Compute AUPRO. Only run with test dataset.'''
np.save('MVTEC3D_amaps.npy', np.array(amaps, dtype=object), allow_pickle=True)
np.save('MVTEC3D_gt.npy', np.array(gtm, dtype=object), allow_pickle=True)

all_fprs, all_pros = compute_pro(anomaly_maps=amaps, ground_truth_maps=gtm)
            
integration_limit = 0.3
    
au_pro = trapezoid(all_fprs, all_pros, x_max=integration_limit)
au_pro /= integration_limit
print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro}")
'''End AUPRO.'''

# Get Residuals For SegAD
torch.save(residuals, os.path.join(f"residual_MVTEC_test.pt"))
print("Done Test!")
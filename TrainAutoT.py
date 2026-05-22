import torch
import torch.nn as nn
from VQGAN_LEDGM import VQGAN_LEDGM
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
from AutoT import AutoregressiveTransformer
import torch.optim as optim
from lr_schedulers import Scheduler_LinearWarmup_CosineDecay
from torch.utils.data import DataLoader
import numpy as np
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""Prepare Datasets."""
"""CableInspect"""
# train = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allcables01_train.pt'), weights_only=True).to(torch.float32) # [num_images, 3, H, W]

# # Create validation set, 10% of training data becomes validation
# keep = torch.ones(2159, dtype=torch.bool)
# keep[0:22] = keep[158:180] = keep[414:436] = keep[603:625] = keep[804:826] = keep[1036:1057] = keep[1216:1238] = keep[1750:1772] = keep[2006:2028] = False   

# val = train[~keep]
# train = train[keep]

"""MVTEC3D."""
train = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_train.pt'), weights_only=True).to(torch.float32)
validation = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_validation.pt'), weights_only=True).to(torch.float32)

# Convert to between [-1 and 1] for LPIPS
train = 2 * train - 1
val = 2 * validation - 1

compression_factor = 8
LATENT_SPACE_HEIGHT = int(train.shape[2] / compression_factor)
LATENT_SPACE_WIDTH = int(train.shape[3] / compression_factor)

"""Prepare Models."""
"""Load Pre-trained VQGAN weights for minimum encoding indices."""
with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
vql_params = config['vqganledgm']

savedir = f"28latentspace_{vql_params['codebook_size']}_5degrees_rgbxyz"

vqgan_info = f"{vql_params['emb_dim']}_{vql_params['z_channels']}_{vql_params['codebook_size']}_{vql_params['beta_vq']}_{vql_params['discriminator_threshold']}"

model = VQGAN_LEDGM(*vql_params.values(), device=device, double=True)
checkpoint = torch.load(os.path.join(os.getcwd(), "Model", f"VQGAN_{savedir}_epoch1500_codedim{vql_params['emb_dim']}_zchan{vql_params['z_channels']}_csize{vql_params['codebook_size']}_disc_threshold{vql_params['discriminator_threshold']}.pt"))
model.load_state_dict(checkpoint['vqgan_state_dict'])
model.eval()

"""Autoregressive transformer."""
autot_params = config['transformer']
max_seq_len = LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH + 1

autot_details = f"{autot_params['num_blocks']}blocks_{autot_params['n_heads']}attnheads_{autot_params['d_model']}d_model"

autot = AutoregressiveTransformer(autot_params['num_blocks'], vql_params['codebook_size'], max_seq_len, autot_params['n_heads'], autot_params['d_model'], device=device)
autot_optim = optim.AdamW(autot.parameters(), lr=1e-6, betas=(0.95, 0.9995), eps=1e-5, weight_decay=0.1)

# Load checkpoint if necessary for longer training
# checkpoint = torch.load(os.path.join(os.getcwd(), "AutoTWeights", f"autot_{savedir}_autot{autot_details}_vqgan{vqgan_info}_50epochs.pt"))
# autot.load_state_dict(checkpoint['autot_state_dict'])
# autot_optim.load_state_dict(checkpoint['optim_state_dict'])

"""Set up for training loop."""
batch_size = 10
epochs = 50
degrees = 10
reverse = False # Simple reverse ordering of the latent space when training a second autoregressive transformer
r_string = "Reverse" if reverse else ""

train_dataset = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=False)
val_dataset = DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=False)

warmup_steps = 500
max_steps = epochs * len(train_dataset)

# Increased over the first 2000 updates
cosine_sched = optim.lr_scheduler.LambdaLR(autot_optim, Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, 0))

cross_entropy = nn.CrossEntropyLoss(reduction='none')
BOS_TOKEN = torch.full((batch_size, 1), vql_params['codebook_size'], device=device)

nll_threshold_99 = - torch.log(torch.Tensor([0.99])).to(device)
nll_threshold_95 = - torch.log(torch.Tensor([0.95])).to(device)

print(f"Threshold 95%: {nll_threshold_95}")
print(f"Threshold 99%: {nll_threshold_99}")

def training_loop(dataset, train=True, reverse=False):
    total_l = 0
    list_val_nll_per_image = []
    for idx, x in enumerate(dataset):
        x = x.to(device)
        
        with torch.no_grad():
            _, _, min_encoding_indices = model.encode(x, degrees)
                
        min_encoding_indices = torch.reshape(min_encoding_indices, (-1, LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH))
        
        if reverse:
            min_encoding_indices = torch.flip(min_encoding_indices, dims=[-1])
        
        input_seq = torch.cat([BOS_TOKEN[:min_encoding_indices.shape[0], :], min_encoding_indices], dim=-1)
        
        if train:
            autot_optim.zero_grad()
            
        choices = autot(input_seq)
        
        nll_per_index = cross_entropy(choices[:, 1:, :].reshape((-1, vql_params['codebook_size'])), min_encoding_indices.reshape(-1))
        nll_per_index = nll_per_index.reshape(choices.shape[0], -1) # [batch, LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH]
        nll_per_image = nll_per_index.mean(dim=-1) # [batch, 1]
        
        if not train:
            list_val_nll_per_image.extend(nll_per_image.cpu().tolist())
        
        loss = nll_per_image.mean()
        
        if train:
            loss.backward()
            
        total_l += loss.detach().item()
        
        if train:
            autot_optim.step()
            cosine_sched.step()
            
            if idx % 150 == 0:
                print(f"Loss: {total_l / (idx + 1)}")
                
    if not train: 
        val_nlls = np.array(list_val_nll_per_image)   # shape: (num_val_images,)
        val_nlls = torch.from_numpy(val_nlls).to(device)
        
        print("Threshold 95: " + str((val_nlls < nll_threshold_95).sum()))
        print("Threshold 99: " + str((val_nlls < nll_threshold_99).sum()))
    
    return total_l

for epoch in range(epochs):
    """-----Training-----"""
    save = True
    total_l = training_loop(train_dataset, True, reverse)
                
    if epoch % 5 == 0 and epoch > 1 and save:
        save = False
        torch.save({'autot_state_dict': autot.state_dict(), 'optim_state_dict': autot_optim.state_dict()}, os.path.join(os.getcwd(), "AutoTCheckPoints", f"{r_string}autot_{savedir}_autot{autot_details}_vqgan{vqgan_info}_{epoch}epochs.pt"))
            
    avg_loss = total_l / len(train_dataset)
    print(f"Epoch {epoch + 1} |{r_string} AutoT Loss: {avg_loss}")
    
    """-----Validation-----"""
    with torch.no_grad():
        total_l = training_loop(val_dataset, False, reverse)
       
        avg_loss = total_l / len(val_dataset)
        print(f"{r_string} Validation | Loss: {avg_loss}")
    
torch.save({'autot_state_dict': autot.state_dict(), 'optim_state_dict': autot_optim.state_dict()}, os.path.join(os.getcwd(), "AutoTWeights", f"{r_string}autot_{savedir}_autot{autot_details}_vqgan{vqgan_info}_{epochs}epochs.pt"))
            



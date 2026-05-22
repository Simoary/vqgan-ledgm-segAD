from VQGAN_LEDGM import VQGAN_LEDGM
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
vql_params = config['vqganledgm']
savedir = "VQVAE" #f"28latentspace_{vql_params['codebook_size']}_5degrees_rgbxyz"
rgbxyz, double = False, False

vqgan_info = f"{vql_params['emb_dim']}_{vql_params['z_channels']}_{vql_params['codebook_size']}_{vql_params['beta_vq']}_{vql_params['discriminator_threshold']}"

"""InspectCable-AD Dataset"""
# Train: 198 for C01 (990). 336 for C02 (1680). 252 for C03 (1260). = 3930
# Move train to Test 20% => 198 for C01, 336 for C02, 252 for C03. 
# Anomaly: 188 for C01 (940). 204 for C02 (1020). 233 for C03 (1165). = 3126

# Dataset already in [-1, 1] for LPIPS
# train = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allcables01_train.pt'), weights_only=True).to(torch.float32) # [2159, 3, 384, 768]

# pass_id_one = torch.cat([torch.zeros(158), torch.ones(255), torch.ones(190) * 2])
# pass_id_two = torch.cat([torch.zeros(201), torch.ones(232), torch.ones(180) * 2])
# pass_id_tri = torch.cat([torch.zeros(534), torch.ones(256), torch.ones(153) * 2])

# labels = torch.cat([pass_id_one, pass_id_two, pass_id_tri])

# Set aside for test dataset
# Indices 0-197 for c01, 990-1326 for C02, 2670-2922 for c03 
# keep = torch.ones(3930, dtype=torch.bool)
# keep[0:198]       = False   
# keep[990:1326]    = False   
# keep[2670:2922]   = False   

# labels = labels[keep]
# train = train[keep]
"""End CableInspect."""

"""3D MVTEC Dataset"""
labels = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_train_labels.pt'), weights_only=True) # [2656, 1]
train = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_train.pt'), weights_only=True).to(torch.float32) # [2656, 6, 224, 224]

# Convert to between [-1 and 1] for LPIPS
train = 2 * train - 1
rgbxyz, double = True, True
"""End 3D."""

ground_truth = torch.nn.functional.one_hot(torch.flatten(labels).to(torch.int64)).to(torch.float32)

batch_size = 8
epochs = 1500
degrees = 10
warmup_epochs = 30

dataset = DataLoader(TensorDataset(train, ground_truth), batch_size=batch_size, shuffle=True, pin_memory=False)

model = VQGAN_LEDGM(*vql_params.values(), device=device, warmup_epochs=warmup_epochs, rgbxyz=rgbxyz, double=double) # Double channels for Decoder if training on MVTec3D
# model.load_pretrained_weights(os.path.join(os.getcwd(), "Model", f"VQGAN_{savedir}_epoch1100_codedim{vql_params['emb_dim']}_zchan{vql_params['z_channels']}_csize{vql_params['codebook_size']}_disc_threshold0.pt"))                        
model.train_model(dataset, epochs, savedir, degrees)
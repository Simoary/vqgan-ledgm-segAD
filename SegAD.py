import numpy as np
from scipy.stats import skew, kurtosis
import os
from VQGAN_LEDGM import VQGAN_LEDGM
from AutoT import AutoregressiveTransformer
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

"""
K is the number of pixel-wise anomaly detectors
L mutually exclusive segments. In CableInspect-AD, we have two segments, the cable and the background, masked by Segment Anything Model (SAM)

Final feature vector f:
- q_k, vector of L (for each segment) numbers where each stands for a 99.5% quantile from anomaly map f_k(I) over pixels where s_l(x) = 1
- For every anomaly map f_k(I): (all size L)
        -> we calculate a vector for skew z_k.  skew = (mean - median) / standard deviation (measure of asymmetry)
        -> kurtosis c_k. kurtosis = E[(X - mean)^4] / (standard deviaiton)^4 (measures the tails of the distribution, are ther more outliers or closer to the middle)
        -> mean m_k.
        
Therefore, the length of feature vector f is K * L * 4 + 1 (+ 1 for supervised classifier score) concatenated together.
"""
    
def compute_statistics(anomaly_maps: list[np.ndarray], seg_mask: np.ndarray, classifier_score: list[float] | None = None):
    """Compute the necessary statistics for the feature vector f to input into the BRF.

    Args:
        anomaly_maps (list[np.ndarray]): List of K anomaly score maps. Pixel-wise outputs from unique anomaly detectors.
        seg_mask (np.ndarray): Integer label map (0 to L-1), each mutually exclusive segment has its own segment class (0 for bakground, 1 for cable, etc...) 
        classifier_score (float | None, optional): Optional classifier score. Defaults to None.
    
    Returns:
        Feature vector f concatenated with the statistics (99.5% quantile, mean, skew, kurtosis, classifier_score)
    """
    
    K = len(anomaly_maps)
    L = int(seg_mask.max()) + 1
    
    features = [classifier_score.tolist()] if classifier_score != None else []
    
    for k in range(K):
        a_map = anomaly_maps[k]
        q_vec = []
        m_vec= []
        z_vec = []
        c_vec = []
        
        for l in range(L):
            mask = (seg_mask == l)
            if mask.sum() == 0:
                q_vec.append(0.0)
                m_vec.append(0.0)
                z_vec.append(0.0)
                c_vec.append(0.0)
                continue
            
            values = a_map[mask].flatten()
            
            q_vec.append(np.quantile(values, 0.995))
            z_vec.append(skew(values))
            c_vec.append(kurtosis(values))
            m_vec.append(torch.mean(values))
            
        features.extend([q_vec, z_vec, c_vec, m_vec])
    
    return np.concatenate(features).astype(np.float32)

def acquire_normal_indices(normal):
    """Function for CableInspectAD dataset. Need to create a training paradigm for the BRF, taking a subset of normal and putting into test."""
    keep = torch.ones(2159, dtype=torch.bool)
    keep[0:52] = keep[158:210] = keep[414:466] = keep[603:655] = keep[804:856] = keep[1036:1087] = keep[1216:1268] = keep[1750:1802] = keep[2006:2058] = False   

    for_test = normal[~keep]
    for_train = normal[keep]
    
    return for_train, for_test

def acquire_anomalous_indices(anomalies):
    """Function for CableInspectAD dataset. Need to create a training paradigm for the BRF, taking a subset of anomalies and putting into train."""
    keep = torch.ones(976, dtype=torch.bool)
    keep[0:60] = keep[274:349] = keep[617:677] = False

    for_test = anomalies[~keep]
    for_train = anomalies[keep]
    
    return for_train, for_test
    
def display_images(data, segmasks, anomalymaps):
    nrows = 3
    ncols = 3

    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9))

    data = (data + 1) / 2

    for i in range((min(nrows, batch_size))):
        for u in range(ncols):
            if u == 0:
                axes[i][u].imshow(data[i].cpu().permute((1, 2, 0)))
                axes[i][u].set_title('Train')
            elif u == 1:
                axes[i][u].imshow(segmasks[i].cpu().permute((1, 2, 0)))
                axes[i][u].set_title('SegMask')
            elif u == 2:
                axes[i][u].imshow(anomalymaps[i].cpu().permute(1, 2, 0))
                axes[i][u].set_title('Ground Truth')            
            axes[i][u].axis('off')
        
    plt.tight_layout() # better spacing
    plt.show()

def acquire_feature_vectors(data, vql, transformer, transformer_reverse):
    
    # EDIT HERE WITH NUM_CLASSES + 1
    f = torch.zeros((len(data), 1 * 2 * 4 + 12)) # K * L * 4 + 4, our case y_classes is 10 + nll_per_image = 12
    for i, (imgs, segmasks, anomalymaps) in enumerate(data):
        with torch.no_grad():
            _, _, min_encoding_indices = vql.encode(imgs.to(device), 0)
            
            image_order = min_encoding_indices.view(batch_size, LATENT_SPACE_HEIGHT, LATENT_SPACE_WIDTH)
            z_q = vql.indices_to_z_q(image_order) # [batch, emb_dim, LATENT_SPAC_HEIGHT, LATENT_SPACE_WIDTH]
            
            '''LEDGM score.'''
            y = vql.classify(z_q)
            ledgm_score = torch.softmax(y, dim=-1)
            
            '''Autoregressive Transformer NLL per image score.'''
            indices = torch.reshape(min_encoding_indices, (-1, LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH))
            
            input_seq = torch.cat([BOS_TOKEN[:indices.shape[0], :], indices], dim=-1) # [batch, LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH + 1]
    
            choices = transformer(input_seq) # [batch, LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH + 1, codebook_size]
            
            
            nll_per_index = cross_entropy(choices[:, 1:, :].reshape((-1, vql_params['codebook_size'])), indices.reshape(-1)) # NLL per index
            nll_per_index = nll_per_index.reshape(choices.shape[0], -1)
            nll_per_image = nll_per_index.mean(dim=-1)
            
            # For Reverse, should make a new function for this
            indices = torch.flip(indices, dims=[-1])
            input_seq = torch.cat([BOS_TOKEN[:indices.shape[0], :], indices], dim=-1)
            choices_reverse = transformer_reverse(input_seq)
            
            nll_per_index = cross_entropy(choices_reverse[:, 1:, :].reshape((-1, vql_params['codebook_size'])), indices.reshape(-1)) # NLL per index
            nll_per_index = nll_per_index.reshape(choices.shape[0], -1)
            reverse_nll_per_image = nll_per_index.mean(dim=-1)
            
            '''Statistical features of segmentation masks and anomaly maps.'''
            c_scores = torch.cat([ledgm_score.squeeze(0), nll_per_image, reverse_nll_per_image], dim=-1)
            feature_vector = compute_statistics(anomalymaps, segmasks.squeeze(0), c_scores)
            
            f[i] = torch.tensor(feature_vector)

    return f


"""
To train the Boosted Random Forest (BRF), we need the anomaly maps, which mean the residual = (x - healthy_recon) of all K transformers.
We need then the segmentation maps for every image (normal and anomalous)
We need the classifier score from the LEDGM.
Acquire feature vector f from the anomaly maps and segmentation maps from function compute_features above.

Train BRF classifier on normal and anomalous data. Then we have it for inference for the anomaly score
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load hyperparameters
with open("config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    
'''CableInspect'''
# # Images
# normal = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allcables01_train.pt'), weights_only=True).to(torch.float32)
# anomalies = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'allcables01_test.pt'), weights_only=True).to(torch.float32)

# # Segmentation Masks
# segmasks_normal = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'segmasks_normal.pt'), weights_only=True).to(torch.float32)
# segmasks_anomalous = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'segmasks_anomalous.pt'), weights_only=True).to(torch.float32)

# # Anomaly Maps
# anomalymaps_normal = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'residual_train.pt'), weights_only=True).to(torch.float32)
# anomalymaps_anomalous = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'residual_test.pt'), weights_only=True).to(torch.float32)

'''MVTEC3D'''
# Images
normal = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_train.pt'), weights_only=True).to(torch.float32) # [2656, 6, 224, 224]
anomalies = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'MVTEC3D_test.pt'), weights_only=True).to(torch.float32) # [1197, 6, 224, 224]
print(normal.shape)

# Segmentation Masks
segmasks_normal = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'segmasks_normal_mvtec.pt'), weights_only=True).to(torch.float32)
segmasks_anomalous = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'segmasks_anomalous_mvtec.pt'), weights_only=True).to(torch.float32)
print(segmasks_normal.shape)

# Anomaly Maps
anomalymaps_normal = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'residual_MVTEC_train.pt'), weights_only=True).to(torch.float32)
anomalymaps_anomalous = torch.load(os.path.join(os.getcwd(), 'ObjectTensors', f'residual_MVTEC_test.pt'), weights_only=True).to(torch.float32)
print(anomalymaps_normal.shape)


compression_factor = 8
LATENT_SPACE_HEIGHT = int(normal.shape[2] / compression_factor)
LATENT_SPACE_WIDTH = int(normal.shape[3] / compression_factor)
max_seq_len = LATENT_SPACE_HEIGHT * LATENT_SPACE_WIDTH + 1

'''1. Parameters for VQGAN-LEDGM.'''
vql_params = config['vqganledgm']
savedir = f"28latentspace_{vql_params['codebook_size']}_5degrees_rgbxyz"

vqgan_info = f"{vql_params['emb_dim']}_{vql_params['z_channels']}_{vql_params['codebook_size']}_{vql_params['beta_vq']}_{vql_params['discriminator_threshold']}"

vql = VQGAN_LEDGM(*vql_params.values(), device=device, double=True)
vql.load_pretrained_weights(os.path.join(os.getcwd(), "Model", f"VQGAN_{savedir}_epoch1500_codedim{vql_params['emb_dim']}_zchan{vql_params['z_channels']}_csize{vql_params['codebook_size']}_disc_threshold{vql_params['discriminator_threshold']}.pt"))
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


# display_images(train, segmasks_normal, anomalymaps_normal)
# display_images(test, segmasks_anomalous, anomalymaps_anomalous)

"""Setting up train and test datasets. Above is for Cables, Below is for MVTec3D."""
# Remember, train has only normals and testing has only anomalies
# To train the BRF, we need a supervised paradigm, a dataset with both normal and anomalous data
'''Original Images.'''
# normal_train, normal_test = acquire_normal_indices(normal)
# anomalies_train, anomalies_test = acquire_anomalous_indices(anomalies)

# brf_train = torch.cat([normal_train, anomalies_train], dim=0)
# brf_test = torch.cat([normal_test, anomalies_test], dim=0)

brf_train = normal
brf_test = anomalies

'''Segmentation Masks.'''
# segmasks_normal_train, segmasks_normal_test = acquire_normal_indices(segmasks_normal)
# segmasks_anomalous_train, segmasks_anomalous_test = acquire_anomalous_indices(segmasks_anomalous)

# segmasks_train = torch.cat([segmasks_normal_train, segmasks_anomalous_train], dim=0)
# segmasks_test = torch.cat([segmasks_normal_test, segmasks_anomalous_test], dim=0)

segmasks_train = segmasks_normal
segmasks_test = segmasks_anomalous

'''Anomaly Maps.'''
# anomalymaps_normal_train, anomalymaps_normal_test = acquire_normal_indices(anomalymaps_normal)
# anomalymaps_anomalous_train, anomalymaps_anomalous_test = acquire_anomalous_indices(anomalymaps_anomalous)

# anomalymaps_train = torch.cat([anomalymaps_normal_train, anomalymaps_anomalous_train], dim=0)
# anomalymaps_test = torch.cat([anomalymaps_normal_test, anomalymaps_anomalous_test], dim=0)

anomalymaps_train = anomalymaps_normal
anomalymaps_test = anomalymaps_anomalous 

"""Training Loop for SegAD."""
# labels_normal_train = torch.zeros((1692,))
# labels_anomalies_train = torch.ones((781,))
# labels_train = torch.cat([labels_normal_train, labels_anomalies_train], dim=0)

# labels_normal_test = torch.zeros((467,))
# labels_anomalies_test = torch.ones((195,))
# labels_test = torch.cat([labels_normal_test, labels_anomalies_test], dim=0)

"""
Calculate statistics with above function.
Save feature vector f statistics on disk.
Train Boosted Random Forest (BRF) for anomaly score.
"""
cross_entropy = nn.CrossEntropyLoss(reduction='none')
batch_size = 1
BOS_TOKEN = torch.full((batch_size, 1), vql_params['codebook_size'], device=device)
train = DataLoader(TensorDataset(brf_train, segmasks_train, anomalymaps_train), batch_size=batch_size, shuffle=False, pin_memory=False)
test = DataLoader(TensorDataset(brf_test, segmasks_test, anomalymaps_test), batch_size=batch_size, shuffle=False, pin_memory=False)


feature_vectors = acquire_feature_vectors(train, vql, autot, autot_reverse)
torch.save(feature_vectors, os.path.join(os.getcwd(), "ObjectTensors", f"MVTECBRF_featurevectors_train.pt"))

print("Done Training!")

feature_vectors = acquire_feature_vectors(test, vql, autot, autot_reverse)
torch.save(feature_vectors, os.path.join(os.getcwd(), "ObjectTensors", f"MVTECBRF_featurevectors_test.pt"))

print("Done Testing!")
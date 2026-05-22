import torch.nn as nn
from VQModels import Encoder, Decoder, VectorQuantization, Classifier

class VQAuto(nn.Module):
    
    def __init__(self, img_channels, z_channels, n_emb, emb_dim, num_classes, beta_vq=0.25, double=False):
        """Initialise Autoencoder portion of VQGAN."""
        super(VQAuto, self).__init__()
        
        self.encode = Encoder(img_channels, z_channels)
        self.vq = VectorQuantization(n_emb, emb_dim, beta_vq)
        self.decode = Decoder(img_channels, z_channels, double)
        self.pre_quant_conv = nn.Conv2d(z_channels, emb_dim, 1)
        self.post_quant_conv = nn.Conv2d(emb_dim, z_channels, 1)
        
        self.q_classify = Classifier(emb_dim, num_classes)
        
    def encode_to_quantization(self, x):
        """Helper function for use later for the autoregressive transformers. Get the Encoding Indices of the image."""
        z = self.encode(x)
        
        pre_quant_z = self.pre_quant_conv(z)
        
        z_q, _, min_encoding_indices = self.vq(pre_quant_z)
        
        return z, z_q, min_encoding_indices
    
    def return_quantised_vectors(self, indices):
        return self.vq.get_quantized_vectors(indices)
    
    def quantised_to_reconstruction(self, z_q):
        post_quant_z = self.post_quant_conv(z_q)
        
        return self.decode(post_quant_z)
    
    def classifier_y(self, z_q):
        return self.q_classify(z_q)
        
    def forward(self, x):
        """Standard Forward from image x to reconstruction/classification."""
        z = self.encode(x)
        
        pre_quant_z = self.pre_quant_conv(z)
        
        z_q, vq_loss, _ = self.vq(pre_quant_z)
        
        y_classes = self.q_classify(z_q)
        
        post_quant_z_q = self.post_quant_conv(z_q)
        
        x_recon = self.decode(post_quant_z_q)
        
        return x_recon, vq_loss, y_classes
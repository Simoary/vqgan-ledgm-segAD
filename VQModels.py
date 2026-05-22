import torch
import torch.nn as nn

class GroupNorm(nn.Module):
    def __init__(self, channels, num_groups = 32):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
        
    def forward(self, x):
        return self.gn(x)

class Swish(nn.Module):
    def forward(self, a):
        return a * torch.sigmoid(a)
    
class Block(nn.Module):
    """A simple Block for the ResnetBlock."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, num_groups=32):
        super(Block, self).__init__()
            
        self.block = nn.Sequential(
                GroupNorm(in_channels, num_groups),
                Swish(),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
            )
            
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    """A Resnet Block for the VQGAN."""
    
    def __init__(self, in_channels, out_channels, stride=1, num_groups=32):
        """Constructor for Resnet Block."""
        super(ResnetBlock, self).__init__()
        
        self.block1 = Block(in_channels, out_channels, stride=stride, num_groups=num_groups)
        self.block2 = Block(out_channels, out_channels, num_groups=num_groups)
        
        if in_channels != out_channels: # When changing channels, use a stride of 2
            self.x_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.x_skip = None
    
    def forward(self, x):
        """Forward propogation through the network. Most promising Results => Convolutions, Normalize, Activation."""
        h = self.block1(x) # GroupNorm, Activation, Conv2d
        residual = self.block2(h)
        
        if self.x_skip: # A need when the number of channels has changed
            x = self.x_skip(x)
            
        # x + F(x) (Residual)
        return x + residual 
    
class Downsample(nn.Module):
    """Downsample for Encoder. Simple Pooling layer."""
    
    def __init__(self, channels):
        super(Downsample, self).__init__()
        
        self.downsample = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)
        
    def forward(self, x):
        # Asymmetric padding
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        # Downsampling directly with convolutional layers that have a stride of 2
        return self.downsample(x)
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = GroupNorm(in_channels)
        
        # Query, Key and Value weight matrices for attention
        # dot-product between the query and key vectors, these two vectors have to contain the same number of elements (dq = dk)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
                                 
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class Encoder(nn.Module):
    """Encoder of the VQGAN. Receives as input an image, and outputs a latent fector z, which will be used for quantization."""
    
    def __init__(self, img_channels, z_channels):
        """Constructor for Encoder."""
        super(Encoder, self).__init__()
        # 128, 128, 128, 256, 256, 512
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [28]
        num_res_blocks = 2
        resolution = 224
        model = [nn.Conv2d(img_channels, 128, 3, 1, 1)]
        # Building the Encoder 
        for channel_idx in range(len(channels) - 1):
            in_channels = channels[channel_idx]
            out_channels = channels[channel_idx + 1]
            for _ in range(num_res_blocks):
                # More than 1 resnetblock in a row is necessary, 1 resnetblock is simply a linear layer.
                model.append(ResnetBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    model.append(AttnBlock(in_channels))
            if (channel_idx != len(channels) - 2) and (channel_idx != len(channels) - 4):
                model.append(Downsample(channels[channel_idx + 1]))
                resolution //= 2
        
        model.append(ResnetBlock(channels[-1], channels[-1]))
        model.append(AttnBlock(channels[-1]))
        model.append(ResnetBlock(channels[-1], channels[-1]))
        model.append(GroupNorm(channels[-1]))
        model.append(Swish())
        model.append(nn.Conv2d(channels[-1], z_channels, 3, 1, 1))
                    
        self.encode = nn.Sequential(*model)
        
    def forward(self, x):
        """Forward propogation through the network. Downsample then resblocks."""
        z = self.encode(x)
        return z
        
class VectorQuantization(nn.Module):
    """Vector Quantized Latent Space in the VQGAN."""

    def __init__(self, n_emb, emb_dim, beta_vq=0.25):
        """Initialize Embedding Codebook."""
        super(VectorQuantization, self).__init__()
        self.emb_dim = emb_dim
        
        # nn.Embedding, lookup table like a linear layer
        self.embedding = nn.Embedding(n_emb, emb_dim)  
        self.embedding.weight.data.normal_()
        
        self.beta_vq = beta_vq
        
    def get_quantized_vectors(self, indices):
        b, h, w = indices.shape
        
        z_q = self.embedding(indices).view((b, h, w, -1))

        z_q = torch.nn.functional.normalize(z_q, p=2, dim=-1)
        z_q = z_q.permute(0, 3, 1, 2)
        return z_q
        
    def get_rotation(self, e_hat, q_hat, e):
        r = ((e_hat + q_hat) / torch.norm(e_hat + q_hat, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, r.unsqueeze(-1)), r.unsqueeze(1)) + 2 * torch.bmm(
        torch.bmm(e, e_hat.unsqueeze(-1).detach()), q_hat.unsqueeze(1).detach())
        return e

    def forward(self, z):
        """Perform Quantization."""
        z = z.permute(0, 2, 3, 1).contiguous()
        
        # l2 Norm comparison. Further improves training stability and reconstruction quality
        z_flat_norm = torch.nn.functional.normalize(z.view(-1, self.emb_dim), p=2, dim=-1)
        codebook_norm = torch.nn.functional.normalize(self.embedding.weight, p=2, dim=-1)
        
        d = torch.sum(z_flat_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook_norm ** 2, dim=1) - \
            2*(torch.matmul(z_flat_norm, codebook_norm.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        z_q, z = torch.nn.functional.normalize(z_q, p=2, dim=-1), torch.nn.functional.normalize(z, p=2, dim=-1)
        
        loss = torch.mean((z_q - z.detach())**2) + self.beta_vq * torch.mean((z_q.detach() - z)**2)

        """1. Gradients need to be passed through to the encoder using straight-through 
        estimator, since we argmin index the embedding codebook"""
        # z_q = z + (z_q - z).detach()
        
        """2. Another way to pass gradients to encoder: Rotation Trick"""
        b, h, w, c = z_q.shape
        z = z.view(-1, self.emb_dim)
        z_q = z_q.view(-1, self.emb_dim)
        
        # Calculating z_q = lambdaRe
        Re = self.get_rotation(z, z_q, z.unsqueeze(1)).squeeze(1)
        
        z_q = Re * (torch.norm(z_q, dim=-1, keepdim=True) / (torch.norm(z, dim=1, keepdim=True) + 1e-6)).detach()
        
        z_q = z_q.view(b, h, w, c)
        """End of Rotation Trick."""
        
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, loss, min_encoding_indices

class Upsample(nn.Module):
    """Upsample for Decoder. Simple Interpolation Layer."""
    def __init__(self, in_channels):
        super(Upsample, self).__init__()

        # Convolution after upsample
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x

class Decoder(nn.Module):
    """Decoder of the VQGAN. receives as input a set of quantized vectors, and outputs the reconstructed image x_recon."""

    def __init__(self, img_channels, z_channels, double=False):
        """Initialisation of the Decoder network."""
        super(Decoder, self).__init__()
        
        channels = [512, 256, 256, 128, 128]
        if double: # For MVTec3D, since we are basically trying to reconstruct 2 images
            channels = [x * 2 for x in channels]
        attn_resolutions = [28]
        num_res_blocks = 3
        resolution = 28

        in_channels = channels[0]
        model = [nn.Conv2d(z_channels, in_channels, 3, 1, 1),
                  ResnetBlock(in_channels, in_channels),
                  AttnBlock(in_channels),
                  ResnetBlock(in_channels, in_channels)]
        
        for i in range(len(channels)):
            out_channels = channels[i]
            for _ in range(num_res_blocks):
                model.append(ResnetBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    model.append(AttnBlock(in_channels))
            if i != 0 and i != 2:
                model.append(Upsample(in_channels))
                resolution *= 2
        
        model.append(GroupNorm(in_channels))
        model.append(Swish())
        model.append(nn.Conv2d(in_channels, img_channels, 3, 1, 1))
        self.decode = nn.Sequential(*model)

    def forward(self, z_q_y):
        """Passing through the Decoder network."""        
        x_recon = self.decode(z_q_y)
        return x_recon
    

class Classifier(nn.Module):
    """Classifier q(y | x, z_q). Helps perform image classification, for later use in SegAD."""

    def __init__(self, emb_dim, num_classes=10):
        """Initialise classifier."""
        super(Classifier, self).__init__()

        # z_q (batch, z_channels, H, W) => (batch, num_classes)
        self.cnn = nn.Sequential(
            ResnetBlock(emb_dim, 32),
            nn.Dropout(0.2),
            Downsample(32),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(32, 16),
            Swish(),
            nn.Dropout(0.4),
            nn.Linear(16, num_classes)
        )

    def forward(self, z_q):
        y_classes = self.cnn(z_q)
        y_classes = self.pool(y_classes)
        y_classes = self.head(y_classes)
        return y_classes

class Discriminator(nn.Module):
    """Patch Discriminator in the VQGAN. (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)"""

    def __init__(self, img_channels, disc_threshold, n_layers=3, num_filters_last=64):
        super(Discriminator, self).__init__()
        
        self.disc_threshold = disc_threshold

        model = [nn.Conv2d(img_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            model += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        model.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.discriminate = nn.Sequential(*model)
        
        self.discriminate.apply(self.discriminator_weights_init)
        
    def discriminator_weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
    def enable_weight(self, epoch):
        disc_factor = 0
        if epoch >= self.disc_threshold:
            disc_factor = 1
        return disc_factor

    def forward(self, x_recon):
        patches = self.discriminate(x_recon)
        return patches
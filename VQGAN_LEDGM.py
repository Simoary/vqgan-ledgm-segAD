from VQModels import Discriminator
from VQAuto import VQAuto
import torch
import torch.nn as nn
import torch.optim as optim
import lpips
import os
from torchvision import utils as vutils
import random
from lr_schedulers import Scheduler_LinearWarmup_CosineDecay
import torchvision.transforms.functional as TF

class VQGAN_LEDGM(nn.Module):
    """VQGAN-LEDGM Hybrid Training Regimen."""

    def __init__(self, z_channels: int, n_emb: int, emb_dim: int, beta_vq: float, num_classes: int, img_channels: int, disc_threshold: int, perceptual_weight: float = 0.1,
                adversarial_weight:float = 0.1, device: str | torch.device = "cuda", warmup_epochs:int = 1, rgbxyz: bool = False, double: bool =False):
        super(VQGAN_LEDGM, self).__init__()
        
        self.vqauto = VQAuto(img_channels, z_channels, n_emb, emb_dim, num_classes, beta_vq, double).to(device)
        
        self.perceptual_loss = lpips.LPIPS(net='vgg').requires_grad_(False).to(device)
        self.perceptual_weight = perceptual_weight
        
        self.rgbxyz = rgbxyz
        
        self.discriminate = Discriminator(img_channels, disc_threshold).to(device)
        self.adversarial_weight = adversarial_weight
        
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.set_optimizers()
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.z_channels, self.codebook_size, self.emb_dim, self.disc_threshold = z_channels, n_emb, emb_dim, disc_threshold
         
    def set_optimizers(self):        
        self.vq_optim = optim.AdamW(self.vqauto.parameters(), lr=4.5e-6, betas=(0.9, 0.99), weight_decay=1e-3)
        
        self.disc_optim = optim.AdamW(self.discriminate.parameters(), lr=4.5e-6, betas=(0.9, 0.99), weight_decay=1e-3)
        
    def configure_schedulers(self, warmup_steps: int, max_steps: int, multiplier_min: float = 0):
        self.vq_sched = optim.lr_scheduler.LambdaLR(self.vq_optim, Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, multiplier_min))
        
        self.disc_sched = optim.lr_scheduler.LambdaLR(self.disc_optim, Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, multiplier_min))
        
    def load_pretrained_weights(self, path: str, train: bool = False):
        checkpoint = torch.load(path, weights_only=True, map_location='cuda:0')
        self.load_state_dict(checkpoint['vqgan_state_dict'])
        if train:
            self.vq_optim.load_state_dict(checkpoint['vq_optim_state_dict'])
            self.disc_optim.load_state_dict(checkpoint['disc_optim_state_dict'])
        
    @torch.no_grad()
    def encode(self, x: torch.Tensor, degrees: int = 0):
        if degrees != 0:
            x = self.image_augmentation(x, degrees=random.randint(-degrees, degrees))
        z, z_q, min_encoding_indices = self.vqauto.encode_to_quantization(x)
        return z, z_q, min_encoding_indices
    
    @torch.no_grad()
    def indices_to_z_q(self, min_indices: torch.Tensor):
        return self.vqauto.return_quantised_vectors(min_indices)
    
    @torch.no_grad()
    def decode(self, z_q: torch.Tensor):
        return self.vqauto.quantised_to_reconstruction(z_q)
    
    @torch.no_grad()
    def classify(self, z_q: torch.Tensor):
        return self.vqauto.classifier_y(z_q)
        
    @torch.no_grad()
    def inference(self, x: torch.Tensor):
        x_recon, z_q, _, min_encoding_indices, y_classes = self.vqauto(x)
        return x_recon, z_q, min_encoding_indices, y_classes
        
    def ae_loss(self, x: torch.Tensor, x_recon: torch.Tensor):
        if self.rgbxyz:
            x_rgb, x_xyz = torch.split(x, 3, dim=1)
            x_recon_rgb, x_recon_xyz = torch.split(x_recon, 3, dim=1) # torch.Size([1, 3, 224, 224])
            
            """For RGB Reconstruction"""
            # LPIPS Perceptual Loss which aims to impitation human perception
            perceptual_loss = self.perceptual_loss(x_rgb, x_recon_rgb) 
            perceptual_loss += self.perceptual_loss(x_xyz, x_recon_xyz)
            perceptual_loss *= self.perceptual_weight            
        else:
            perceptual_loss = self.perceptual_loss(x, x_recon) * self.perceptual_weight
        
        # L1 Reconstruction Loss
        recon_loss = torch.abs(x - x_recon)

        return perceptual_loss, recon_loss   
    
    def calculate_adaptive_weight(self, perceptual_recon_loss: torch.Tensor, gan_loss: torch.Tensor):
        last_layer = self.vqauto.decode.decode[-1]
        last_layer_weights = last_layer.weight
        
        perceptual_recon_loss_grads = torch.autograd.grad(perceptual_recon_loss, last_layer_weights, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weights, retain_graph=True)[0]
    
        λ = torch.norm(perceptual_recon_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return λ
    
    def discriminator_loss(self, patches_fake: torch.Tensor, patches_real: None | torch.Tensor = None):
        # Hinge Loss for Discriminator
        fake_loss = - patches_fake.mean() * 2 if patches_real is None else torch.nn.functional.relu(1. + patches_fake).mean() 
        real_loss = 0 if patches_real is None else torch.nn.functional.relu(1. - patches_real).mean()
        
        return (fake_loss + real_loss) / 2, fake_loss, real_loss
    
    def image_augmentation(self, x: torch.Tensor, degrees: int = 0):        
        """Image Augmentation function, to alter images for better generalization"""
        x = TF.rotate(x, degrees)
        return x
        
    def train_model(self, dataset: torch.FloatTensor, epochs: int, savedir: str, degrees: int):
        """
        Training Loop for the VQGAN-LEDGM Hybrid model.

        Parameters: 
            dataset (DataLoader): Input Data. Shape: [Data samples, Channels, Height, Width]
            epochs (int): Number of times for the model to run through the data for optimization.
            savedir (str): Directory to save the progression of reconstructed images.
            degrees (int): Number of degrees to rotate between for data augmentation
        """
        steps_per_epoch = len(dataset)
        warmup_steps = self.warmup_epochs * steps_per_epoch
        training_steps = epochs * steps_per_epoch
        self.configure_schedulers(warmup_steps, training_steps)
        
        for epoch in range(epochs):
            total_vq = total_discrim = total_recon = total_perceptual = 0
            for i, (x, gt) in enumerate(dataset):
                    x = x.to(self.device)
                    gt = gt.to(self.device) 
                    
                    x = self.image_augmentation(x, degrees=random.randint(-degrees, degrees))
                    
                    self.vq_optim.zero_grad()
                    self.disc_optim.zero_grad()
                    
                    x_recon, vq_loss, y_classes = self.vqauto(x)
                    
                    # for loop index 0 is generator part, 1 is discriminator
                    for q in range(2):
                        if q == 0:
                            perceptual_loss, recon_loss = self.ae_loss(x, x_recon)
                            
                            recon_loss = torch.mean(recon_loss)
                            perceptual_loss = torch.mean(perceptual_loss)
                            
                            perceptual_recon_loss = recon_loss + perceptual_loss
                            
                            patches_fake = self.discriminate(x_recon)
                            g_loss, fake_loss, _ = self.discriminator_loss(patches_fake)
                            
                            if epoch % 3 == 0 and i < 5:
                                print(f"Fake Loss: {fake_loss.item()}")
                            
                            y_loss = self.ce_loss(y_classes, gt)
                            
                            d_weight = self.adversarial_weight
                            
                            d_weight *= self.calculate_adaptive_weight(perceptual_recon_loss, g_loss)
                            
                            disc_factor = self.discriminate.enable_weight(epoch)
                            
                            aeloss = perceptual_recon_loss + vq_loss + y_loss + disc_factor * d_weight * g_loss
                            
                            aeloss.backward()
                            self.vq_optim.step()
                            self.vq_sched.step()
                            
                        if q == 1:
                            disc_factor = self.discriminate.enable_weight(epoch)
                            
                            patches_real = self.discriminate(x)
                            patches_fake = self.discriminate(x_recon.detach())
                            
                            disc_loss, fake_loss, real_loss = self.discriminator_loss(patches_fake, patches_real)
                            
                            discriminator_loss = disc_factor * disc_loss
                            
                            if epoch % 3 == 0 and i < 5:
                                print(f"Fake Loss: {fake_loss.item()} Real Loss: {real_loss.item()}")
                            
                            discriminator_loss.backward()
                            self.disc_optim.step()
                            self.disc_sched.step()
                    
                    total_recon += recon_loss.detach().item()
                    total_perceptual += perceptual_loss.detach().item()
                    total_vq += vq_loss.detach().item()
                    total_discrim += discriminator_loss.detach().item()
                    
                    if epoch % 150 == 0 and epoch > 1:
                        checkpoint = {'vqgan_state_dict': self.state_dict(), 'vq_optim_state_dict': self.vq_optim.state_dict(), 'disc_optim_state_dict': self.disc_optim.state_dict()}
                        torch.save(checkpoint, os.path.join(os.getcwd(), "ModelCheckPoints", f"VQGAN_{savedir}_epoch{epoch}_codedim{self.emb_dim}_zchan{self.z_channels}_csize{self.codebook_size}_disc_threshold{self.disc_threshold}.pt"))

                    """Save Images to Disk to keep an eye on model performance"""
                    if epoch % 2 == 0 and i == 0:
                        with torch.no_grad():              
                            if self.rgbxyz:             
                                rgb, xyz = x[:3, :3, :, :], x[0:1, 3:, :, :]
                                recon_rgb, recon_xyz = x_recon[:3, :3, :, :], x_recon[0:1, 3:, :, :]
                                
                                real = torch.cat((rgb, xyz))
                                fake = torch.cat((recon_rgb, recon_xyz))
                                real_fake_images = torch.cat((real, fake))
                            else:
                                real_fake_images = torch.cat([x[0:4, :, :, :], x_recon[0:4, :, :, :]])
                                
                            real_fake_images = (real_fake_images + 1) / 2 # Convert from [-1, 1] to RGB [0, 1]

                            vutils.save_image(real_fake_images, os.path.join(savedir, f"{epoch}_{i}.jpg"), nrow=4)
                    
                    if epoch % 1 == 0 and i < 5:        
                        print(f"Epoch {epoch}: RECON={total_recon/(i+1)} PERC={total_perceptual/(i+1)} VQ_LOSS={total_vq/(i+1)} GAN_LOSS={total_discrim/(i+1)}")
                        print(self.vq_sched.get_last_lr())
            
        
        checkpoint = {'vqgan_state_dict': self.state_dict(), 'vq_optim_state_dict': self.vq_optim.state_dict(), 'disc_optim_state_dict': self.disc_optim.state_dict()}
        torch.save(checkpoint, os.path.join(os.getcwd(), "Model", f"VQGAN_{savedir}_epoch{epochs}_codedim{self.emb_dim}_zchan{self.z_channels}_csize{self.codebook_size}_disc_threshold{self.disc_threshold}.pt"))


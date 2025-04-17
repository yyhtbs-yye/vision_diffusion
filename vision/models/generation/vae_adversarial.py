import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers.models.autoencoders import autoencoder_kl
from vision.nn.discriminators import VAEDiscriminator
from lpips import LPIPS
from copy import deepcopy
from vision.visualizers.basic_visualizer import visualize_comparisons

class VAEWithAdversarial(pl.LightningModule):
    """VAE with adversarial training as a Lightning module."""
    
    def __init__(self, model_config, train_config, validation_config):
        super().__init__()
        self.automatic_optimization = False  # Manual optimization
        self.save_hyperparameters(ignore=['model_config'])

        # Extract config from model_config dictionary
        self.vae_config = model_config['vae_config']
        self.vae_config['in_channels'] = model_config['in_channels']
        self.vae_config['out_channels'] = model_config['in_channels']

        self.disc_config = model_config['disc_config']
        self.disc_config['in_channels'] = model_config['in_channels']
        
        vae_optimizer_config = train_config.get('vae_optimizer', {})
        disc_optimizer_config = train_config.get('disc_optimizer', {})
        # Training params
        self.vae_steps = vae_optimizer_config.get('vae_steps', 1)
        self.disc_steps = disc_optimizer_config.get('disc_steps', 1)
        self.vae_use_ema = vae_optimizer_config.get('use_ema', True)
        self.vae_ema_decay = vae_optimizer_config.get('ema_decay', 0.999)
        self.vae_ema_start = vae_optimizer_config.get('ema_start', 1000)
        
        # Extract learning rates and betas from the config
        self.vae_learning_rate = vae_optimizer_config.get('learning_rate', 0.0001)
        self.disc_learning_rate = disc_optimizer_config.get('learning_rate', 0.0002)
        self.vae_betas = vae_optimizer_config.get('betas', [0.5, 0.999])
        self.disc_betas = disc_optimizer_config.get('betas', [0.5, 0.999])
        self.vae_weight_decay = vae_optimizer_config.get('weight_decay', 0.01)
        self.disc_weight_decay = disc_optimizer_config.get('weight_decay', 0.01)
        # Load models directly from diffusers
        self.vae = autoencoder_kl.AutoencoderKL.from_config(self.vae_config)
        self.disc = VAEDiscriminator.from_config(self.disc_config)

        self.num_vis_samples = validation_config.get('num_vis_samples', 4)

        # Initialize loss functions
        self.perceptual_loss = LPIPS(net='vgg')
        
        # Create EMA model if requested
        if self.vae_use_ema:
            self.vae_ema = deepcopy(self.vae)
            for param in self.vae_ema.parameters():
                param.requires_grad = False

    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.vae_use_ema:
            return
        
        for ema_param, param in zip(self.vae_ema.parameters(), self.vae.parameters()):
            ema_param.data.mul_(self.vae_ema_decay).add_(param.data, alpha=1 - self.vae_ema_decay)
        
        # Update buffers
        for ema_buffer, buffer in zip(self.vae_ema.buffers(), self.vae.buffers()):
            ema_buffer.data.copy_(buffer.data)

    def forward(self, batch_size=None, img_shape=None, return_dict=True):
        if img_shape is None:
            if batch_size is None:
                raise ValueError("Must provide either batch_size or img_shape")
            img_shape = (batch_size, self.disc.config['in_channels'], 
                        self.disc.config.sample_size, self.disc.config.sample_size)
        
        # Calculate latent dimensions
        latent_channels = self.vae.config.latent_channels
        latent_height = img_shape[2] // self.code_downscale
        latent_width = img_shape[3] // self.code_downscale
        latent_shape = (img_shape[0], latent_channels, latent_height, latent_width)
        
        # Generate noise in latent space
        noise = torch.randn(latent_shape, device=self.device, dtype=self.vae.dtype)
        generated_images = self.vae.decode(noise).sample
        
        # Scale images from [-1, 1] to [0, 1], if applicable
        # generated_images = generated_images * 0.5 + 0.5
        
        if return_dict:
            return {'samples': generated_images}
        else:
            return generated_images    
    
    def kl_divergence_loss(self, mu, logvar):
        """Compute KL divergence loss."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    def train_discriminator(self, real_imgs, fake_imgs):
        """Train the discriminator using real and fake images."""
        disc_optimizer = self.optimizers()[1]
        disc_optimizer.zero_grad()

        # Discriminator predictions
        disc_pred_real = self.disc(real_imgs)["logits"]
        disc_pred_fake = self.disc(fake_imgs.detach())["logits"]

        # Compute discriminator loss (WGAN-style)
        disc_loss = (torch.mean(disc_pred_fake) - torch.mean(disc_pred_real)) * 0.01
        self.manual_backward(disc_loss)
        disc_optimizer.step()

        # Log discriminator metrics
        self.log('disc_loss', disc_loss, prog_bar=True)

        return disc_loss
    
    def train_generator(self, real_imgs, recon_imgs, fake_imgs):
        """Train the generator using real, reconstructed, and fake images."""
        vae_optimizer = self.optimizers()[0]
        vae_optimizer.zero_grad()

        # Encode real images to get latents for KL loss
        latents = self.vae.encode(real_imgs).latent_dist
        mu, logvar = latents.mean, latents.logvar

        # Adversarial loss using fake images
        disc_pred_fake_g = self.disc(fake_imgs)["logits"]
        vae_gan_loss = -torch.mean(disc_pred_fake_g) * 0.01

        # Reconstruction loss using reconstructed images
        vae_mse_loss = nn.functional.mse_loss(recon_imgs, real_imgs)

        # KL divergence loss
        var_kld_loss = self.kl_divergence_loss(mu, logvar) * 0.01

        # Perceptual loss using reconstructed images
        var_lpips_loss = self.perceptual_loss(recon_imgs, real_imgs).mean() * 0.5

        # Total generator loss
        vae_total_loss = vae_gan_loss + vae_mse_loss + var_kld_loss + var_lpips_loss

        # Backward and optimize
        self.manual_backward(vae_total_loss)
        vae_optimizer.step()

        # Log generator metrics
        self.log_dict({
            'vae_total_loss': vae_total_loss,
            'vae_gan_loss': vae_gan_loss,
            'vae_mse_loss': vae_mse_loss,
            'var_kld_loss': var_kld_loss,
            'var_lpips_loss': var_lpips_loss
        }, prog_bar=True)

        # Update EMA model
        if self.vae_use_ema and self.global_step >= self.vae_ema_start:
            self._update_ema()

        return vae_total_loss
        
    def training_step(self, batch, batch_idx):
        """Custom training step with real, reconstructed, and fake images."""
        # Extract images from batch
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.shape[0]
        latent_channels = self.vae.config.latent_channels

        # Generate reconstructed images (encode and decode real_imgs)
        latents = self.vae.encode(real_imgs).latent_dist
        z_recon = latents.sample()  # Reparameterization trick
        recon_imgs = self.vae.decode(z_recon).sample

        # Generate fake images from random latents
        z_fake = torch.randn(batch_size, latent_channels, z_recon.size(-2), z_recon.size(-1),
                             device=real_imgs.device, dtype=real_imgs.dtype)
        fake_imgs = self.vae.decode(z_fake).sample

        # Train discriminator with real and fake images
        disc_loss = self.train_discriminator(real_imgs, fake_imgs)

        # Train generator with real, reconstructed, and fake images
        if batch_idx % self.disc_steps == 0:
            for _ in range(self.vae_steps):
                vae_loss = self.train_generator(real_imgs, recon_imgs, fake_imgs)

        return {"disc_loss": disc_loss}


    def validation_step(self, batch, batch_idx):

        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        
        # Generate sample images using the current model
        with torch.no_grad():
            # Generate reconstructions
            codes = self.vae.encode(real_imgs).latent_dist.mean
            recons = self.vae.decode(codes).sample
            img_mse = nn.functional.mse_loss(recons, real_imgs)

            self.log("val/img_mse", img_mse)
            # For the first batch only, create and log visualization samples
            if batch_idx == 0:
                
                # Create comparison visualizations between real and reconstructed images
                cmp_dict = {
                    'real': real_imgs[:self.num_vis_samples],
                    'recon': recons[:self.num_vis_samples]
                }

                # Log the comparison visualizations to the experiment logger
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=cmp_dict,
                    keys=list(cmp_dict.keys()),
                    global_step=self.global_step,
                    wnb=(1, 0),         # Weights and biases configuration
                    prefix='val'            # Prefix for logging
                )

                rcodes_shape = (self.num_vis_samples, self.vae.config['latent_channels'], 
                              codes.size(-2), codes.size(-1))
                
                rcodes = torch.randn(rcodes_shape, device=self.device, dtype=self.vae.dtype)
                generated_imgs = self.vae.decode(rcodes).sample
                # Create and log visualization of purely generated samples
                gen_dict = {'gen': generated_imgs,}

                # Log the generated samples to the experiment logger
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=gen_dict,
                    keys=list(gen_dict.keys()),
                    global_step=self.global_step,
                    wnb=(1, 0),
                    prefix='val'
                )                
        
        return {"val_loss": img_mse}
    
    def configure_optimizers(self):
        """Configure optimizer for the two models."""
        vae_optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=self.vae_learning_rate,
            betas=tuple(self.vae_betas),
            weight_decay=self.vae_weight_decay,
        )

        disc_optimizer = torch.optim.AdamW(
            self.disc.parameters(),
            lr=self.disc_learning_rate,
            betas=tuple(self.disc_betas),
            weight_decay=self.disc_weight_decay,
        )
        
        return [vae_optimizer, disc_optimizer]
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['vae'] = self.vae.state_dict()
        checkpoint['disc'] = self.disc.state_dict()
        """Save EMA model state."""
        if self.vae_use_ema:
            checkpoint['vae_ema'] = self.vae_ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """Load EMA model state."""
        if 'vae' in checkpoint:
            self.vae.load_state_dict(checkpoint['vae'])
        if 'disc' in checkpoint:
            self.disc.load_state_dict(checkpoint['disc'])
        # Load EMA model state if it exists
        if self.vae_use_ema and 'vae_ema' in checkpoint:
            self.vae_ema.load_state_dict(checkpoint['vae_ema'])
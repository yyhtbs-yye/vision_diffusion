import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers.models.autoencoders import autoencoder_kl
from vision.nn.discriminators import VAEDiscriminator
from lpips import LPIPS
from copy import deepcopy

class VAEWithAdversarial(pl.LightningModule):
    """VAE with adversarial training as a Lightning module."""
    
    def __init__(self, model_config, optimizer_config):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])
        
        # Extract config from model_config dictionary
        self.vae_config = model_config['vae_config']
        self.disc_config = model_config['disc_config']
        
        # Training params
        self.generator_steps = optimizer_config.get('generator_steps', 1)
        self.discriminator_steps = optimizer_config.get('discriminator_steps', 1)
        self.use_ema = optimizer_config.get('use_ema', True)
        self.ema_decay = optimizer_config.get('ema_decay', 0.999)
        self.ema_start = optimizer_config.get('ema_start', 1000)
        
        # Extract optimizer configuration from YAML structure
        self.generator_optimizer_config = optimizer_config.get('generator', {}).get('optimizer', {})
        self.discriminator_optimizer_config = optimizer_config.get('discriminator', {}).get('optimizer', {})
        
        # Extract learning rates and betas from the config
        self.learning_rate_generator = self.generator_optimizer_config.get('lr', 0.0001)
        self.learning_rate_discriminator = self.discriminator_optimizer_config.get('lr', 0.0002)
        self.generator_betas = self.generator_optimizer_config.get('betas', [0.5, 0.999])
        self.discriminator_betas = self.discriminator_optimizer_config.get('betas', [0.5, 0.999])
        
        self.automatic_optimization = False  # Manual optimization
        
        # Load models directly from diffusers
        self.vae = autoencoder_kl.AutoencoderKL.from_config(self.vae_config)
        self.discriminator = VAEDiscriminator.from_config(self.disc_config)
        
        # Initialize loss functions
        self.perceptual_loss = LPIPS(net='vgg')
        
        # Create EMA model if requested
        if self.use_ema:
            self.vae_ema = deepcopy(self.vae)
            for param in self.vae_ema.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Forward pass for inference."""
        if self.use_ema and self.global_step >= self.ema_start:
            latents = self.vae_ema.encode(x).latent_dist
            z = latents.sample()
            reconstructions = self.vae_ema.decode(z).sample
        else:
            latents = self.vae.encode(x).latent_dist
            z = latents.sample()
            reconstructions = self.vae.decode(z).sample
        return {
            "reconstructions": reconstructions,
            "latents": z,
            "mu": latents.mean,
            "logvar": latents.logvar
        }
    
    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema:
            return
        
        for ema_param, param in zip(self.vae_ema.parameters(), self.vae.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        
        # Update buffers
        for ema_buffer, buffer in zip(self.vae_ema.buffers(), self.vae.buffers()):
            ema_buffer.data.copy_(buffer.data)
    
    def kl_divergence_loss(self, mu, logvar):
        """Compute KL divergence loss."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    def train_discriminator(self, real_imgs):
        """Train the discriminator."""
        d_opt = self.optimizers()[1]
        d_opt.zero_grad()
        
        # Generate reconstructions
        with torch.no_grad():
            latents = self.vae.encode(real_imgs).latent_dist
            z = latents.sample()
            reconstructions = self.vae.decode(z).sample
        
        # Discriminator predictions
        disc_pred_real = self.discriminator(real_imgs)["logits"]
        disc_pred_fake = self.discriminator(reconstructions.detach())["logits"]
        
        # Compute discriminator loss (WGAN-style)
        d_loss = (torch.mean(disc_pred_fake) - torch.mean(disc_pred_real)) * 0.01
        self.manual_backward(d_loss)
        d_opt.step()
        
        # Log discriminator metrics
        self.log('loss_disc', d_loss, prog_bar=True)
        
        return d_loss
    
    def train_generator(self, real_imgs):
        """Train the generator (VAE)."""
        g_opt = self.optimizers()[0]
        g_opt.zero_grad()
        
        # VAE forward pass
        latents = self.vae.encode(real_imgs).latent_dist
        mu, logvar = latents.mean, latents.logvar
        z = latents.sample()
        reconstructions = self.vae.decode(z).sample
        
        # Discriminator prediction on reconstructions
        disc_pred_fake_g = self.discriminator(reconstructions)["logits"]
        
        # Generator losses
        # Adversarial loss
        gan_loss = -torch.mean(disc_pred_fake_g) * 0.01
        
        # Reconstruction loss (MSE)
        mse_loss = nn.functional.mse_loss(reconstructions, real_imgs)
        
        # KL divergence loss
        kld_loss = self.kl_divergence_loss(mu, logvar) * 0.01
        
        # Perceptual loss
        lpips_loss = self.perceptual_loss(reconstructions, real_imgs).mean() * 0.5
        
        # Total generator loss
        g_loss = gan_loss + mse_loss + kld_loss + lpips_loss
        
        # Backward and optimize
        self.manual_backward(g_loss)
        g_opt.step()
        
        # Log generator metrics
        self.log_dict({
            'loss_gen': g_loss,
            'loss_gan': gan_loss,
            'loss_mse': mse_loss,
            'loss_kld': kld_loss,
            'loss_lpips': lpips_loss
        }, prog_bar=True)
        
        # Update EMA model after generator update
        if self.use_ema and self.global_step >= self.ema_start:
            self._update_ema()
            
        return g_loss
    
    def training_step(self, batch, batch_idx):
        """Custom training step with alternating generator and discriminator updates."""
        # Extract images from batch
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        
        # Train discriminator
        d_loss = self.train_discriminator(real_imgs)
        
        # Train generator (VAE) every discriminator_steps iterations
        if batch_idx % self.discriminator_steps == 0:
            for _ in range(self.generator_steps):
                g_loss = self.train_generator(real_imgs)
        
        return {"loss_disc": d_loss}
    
    def validation_step(self, batch, batch_idx):
        """Validation with visualization of reconstructions."""
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        
        # Use appropriate model for validation
        if self.use_ema and self.global_step >= self.ema_start:
            output = self(real_imgs)  # Uses EMA model in forward
        else:
            latents = self.vae.encode(real_imgs).latent_dist
            z = latents.sample()
            reconstructions = self.vae.decode(z).sample
            output = {
                "reconstructions": reconstructions,
                "mu": latents.mean,
                "logvar": latents.logvar
            }
        
        reconstructions = output["reconstructions"]
        
        # Compute metrics
        mse = nn.functional.mse_loss(reconstructions, real_imgs)
        psnr = self._compute_psnr(reconstructions, real_imgs)
        
        # Log metrics
        self.log('val_mse', mse)
        self.log('val_psnr', psnr)
        
        # Visualize for first batch only
        if batch_idx == 0:
            self._visualize_batch(real_imgs, reconstructions)
        
        return {"val_mse": mse, "val_psnr": psnr}

    def _visualize_batch(self, real_imgs, reconstructions):
        """Visualize reconstructions and noise-generated images in two separate figures."""
        # Normalize images to [0, 1] for visualization
        real_imgs_vis = (real_imgs.clamp(-1, 1) + 1) / 2
        recon_vis = (reconstructions.clamp(-1, 1) + 1) / 2
        
        # Number of images to visualize (up to 8)
        num_images = min(8, real_imgs.size(0))
        
        # Figure 1: Original vs Reconstructed
        comparison_images = []
        for i in range(num_images):
            # Concatenate original and reconstructed images horizontally
            single_row = torch.cat([
                real_imgs_vis[i],  # Original
                recon_vis[i]       # Reconstructed
            ], dim=2)  # Concatenate along width
            comparison_images.append(single_row)
        
        # Stack all rows vertically into a single image for Figure 1
        recon_comparison = torch.cat(comparison_images, dim=1)  # Concatenate along height

        latent_dist = self.vae.encode(recon_vis)
        noise_shape = list(latent_dist['latent_dist'].mean.shape)
        noise_shape[0] = 8
        # Figure 2: Noise-generated images
        # Generate pure noise images
        noise_z = torch.randn(*noise_shape).to(real_imgs.device)
        noise_generated = self.vae.decode(noise_z).sample
        noise_vis = (noise_generated.clamp(-1, 1) + 1) / 2
        
        # Stack noise-generated images vertically
        noise_comparison = torch.cat([noise_vis[i] for i in range(num_images)], dim=2)
        
        # Log both figures to tensorboard
        self.logger.experiment.add_image(
            'val_recon_comparison',
            recon_comparison,
            self.global_step
        )
        self.logger.experiment.add_image(
            'val_noise_generated',
            noise_comparison,
            self.global_step
        )

    
    def _compute_psnr(self, img1, img2):
        """Compute Peak Signal-to-Noise Ratio."""
        mse = nn.functional.mse_loss(img1, img2)
        if mse == 0:
            return torch.tensor(100.0, device=img1.device)
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr
    
    def configure_optimizers(self):
        """Configure optimizers for generator and discriminator."""
        # Get optimizer type from config (default to Adam)
        generator_opt_type = self.generator_optimizer_config.get('type', 'Adam')
        discriminator_opt_type = self.discriminator_optimizer_config.get('type', 'Adam')
        
        # Get optimizer class based on type
        generator_optimizer_class = getattr(torch.optim, generator_opt_type)
        discriminator_optimizer_class = getattr(torch.optim, discriminator_opt_type)
        
        # Create optimizers with parameters from YAML
        g_opt = generator_optimizer_class(
            self.vae.parameters(),
            lr=self.learning_rate_generator,
            betas=tuple(self.generator_betas)
        )
        
        d_opt = discriminator_optimizer_class(
            self.discriminator.parameters(),
            lr=self.learning_rate_discriminator,
            betas=tuple(self.discriminator_betas)
        )
        
        return [g_opt, d_opt]
    
    def on_save_checkpoint(self, checkpoint):
        """Save EMA model state."""
        if self.use_ema:
            checkpoint['vae_ema'] = self.vae_ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """Load EMA model state."""
        if self.use_ema and 'vae_ema' in checkpoint:
            self.vae_ema.load_state_dict(checkpoint['vae_ema'])
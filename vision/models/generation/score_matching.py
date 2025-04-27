import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy

from diffusers.models.unets import UNet2DModel

from vision.visualizers.basic_visualizer import visualize_comparisons

from vision.solvers.ode_solvers import langevin_dynamics, euler_maruyama_sde

class PixelScoreMatchingModel(pl.LightningModule):
    
    def __init__(self, model_config, train_config, validation_config):
        super().__init__()
        # Disable automatic optimization
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=['model_config'])
        
        # Create a UNet Model
        self.unet_config = model_config.get('unet_config', {})
        self.unet_config['in_channels'] = model_config.get('in_channels', 3)  # RGB images
        self.unet = UNet2DModel.from_config(self.unet_config)

        # Define number of timesteps for UNet compatibility
        self.num_timesteps = 1000  # Used for timestep embedding

        # Sampling parameters
        self.num_inference_steps = validation_config.get('num_inference_steps', 50)
        self.num_vis_samples = validation_config.get('num_vis_samples', 4)
        
        # Score matching parameters
        self.sigma_min = train_config.get('sigma_min', 0.01)
        self.sigma_max = train_config.get('sigma_max', 1.0)
        self.sigma_data = train_config.get('sigma_data', 0.5)  # Data noise level
        
        # Training params
        optimizer_config = train_config.get('optimizer', {})
        self.learning_rate = optimizer_config.get('learning_rate', 1e-4)
        self.betas = optimizer_config.get('betas', [0.9, 0.999])
        self.weight_decay = optimizer_config.get('weight_decay', 0.0)
        self.use_ema = optimizer_config.get('use_ema', True)
        self.ema_decay = optimizer_config.get('ema_decay', 0.999)
        self.ema_start = optimizer_config.get('ema_start', 1000)
        
        # EMA model setup
        if self.use_ema:
            self.unet_ema = deepcopy(self.unet)
            for param in self.unet_ema.parameters():
                param.requires_grad = False
        
    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema: return
        
        for ema_param, param in zip(self.unet_ema.parameters(), self.unet.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        
        # Update buffers
        for ema_buffer, buffer in zip(self.unet_ema.buffers(), self.unet.buffers()):
            ema_buffer.data.copy_(buffer.data)
    
    def _sample_noise_levels(self, batch_size):
        """Sample continuous noise levels on a logarithmic scale."""
        u = torch.rand(batch_size, device=self.device)
        log_sigma = torch.log(self.sigma_min) * (1 - u) + torch.log(self.sigma_max) * u
        return torch.exp(log_sigma)
    
    def forward(self, batch_size=None, img_shape=None, num_inference_steps=None, return_dict=True, method='langevin'):
        """Generate samples using score-based sampling."""
        
        
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
            
        if img_shape is None:
            if batch_size is None:
                raise ValueError("Must provide either batch_size or img_shape")
            img_shape = (batch_size, self.unet_config['in_channels'], 
                         self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        
        # Start from random noise
        x = torch.randn(img_shape, device=self.device, dtype=self.unet.dtype)
        
        # Choose sampling method
        if method == 'langevin':
            generated_images = langevin_dynamics(
                x, 
                score_fn=lambda x_t, sigma: self.unet(x_t, self._sigma_to_t(sigma)).sample / sigma.view(-1, 1, 1, 1),
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                num_steps=num_inference_steps
            )
        else:  # Use SDE solver
            generated_images = euler_maruyama_sde(
                x,
                score_fn=lambda x_t, sigma: self.unet(x_t, self._sigma_to_t(sigma)).sample / sigma.view(-1, 1, 1, 1),
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                num_steps=num_inference_steps
            )
        
        if return_dict:
            return {'samples': generated_images}
        else:
            return generated_images
    
    def _sigma_to_t(self, sigma):
        """Convert noise level to timestep for UNet embedding."""
        # Map sigma in [sigma_min, sigma_max] to t in [0, num_timesteps-1]
        log_sigma_ratio = torch.log(sigma / self.sigma_min) / torch.log(self.sigma_max / self.sigma_min)
        t = log_sigma_ratio * (self.num_timesteps - 1)
        return t.long().clamp(0, self.num_timesteps - 1)

    def training_step(self, batch, batch_idx):
        """Train the model to predict scores using denoising score matching."""
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.shape[0]
        
        # Sample noise levels
        sigmas = self._sample_noise_levels(batch_size)
        
        # Sample noise
        noise = torch.randn_like(real_imgs)
        
        # Add noise to the images
        noisy_imgs = real_imgs + sigmas.view(-1, 1, 1, 1) * noise
        
        # Get timesteps for UNet (convert sigma to t)
        timesteps = self._sigma_to_t(sigmas)
        
        # Predict score
        score_pred = self.unet(noisy_imgs, timesteps).sample
        
        # Target score: -noise / sigma (normalized negative noise)
        score_target = -noise / sigmas.view(-1, 1, 1, 1)
        
        # Denoising Score Matching loss with importance sampling
        # Weight by sigma^2 to prevent overemphasis on small noise levels
        loss = F.mse_loss(score_pred, score_target, reduction='none')
        loss = (loss * sigmas.view(-1, 1, 1, 1)**2).mean()
        
        # Manually optimize
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        # Update EMA
        if self.use_ema and self.global_step >= self.ema_start:
            self._update_ema()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validate with single-step reconstruction and sample generation."""
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.size(0)

        with torch.no_grad():
            # Sample random noise levels
            sigmas = self._sample_noise_levels(batch_size)
            timesteps = self._sigma_to_t(sigmas)
            
            # Add noise to the images
            noise = torch.randn_like(real_imgs)
            noisy_imgs = real_imgs + sigmas.view(-1, 1, 1, 1) * noise
            
            # Predict score
            score_pred = self.unet(noisy_imgs, timesteps).sample
            
            # Estimate reconstruction by applying score vector
            # x_approx = x_noisy + sigma^2 * score
            recon_imgs = noisy_imgs + sigmas.view(-1, 1, 1, 1)**2 * score_pred
            
            # Clip reconstructed images to valid range
            recon_imgs = torch.clamp(recon_imgs, -1.0, 1.0)

            # Calculate MSE between original and reconstructed images
            img_mse = F.mse_loss(real_imgs, recon_imgs)
            self.log("val/img_mse", img_mse)

            if batch_idx == 0:
                # Visualization: Real vs. Reconstructed
                cmp_dict = {
                    'real': real_imgs[:self.num_vis_samples], 
                    'recon': recon_imgs[:self.num_vis_samples], 
                }

                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=cmp_dict,
                    keys=list(cmp_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),
                    prefix='val'
                )

                # Generate samples from noise
                noise_shape = (self.num_vis_samples, self.unet_config['in_channels'], 
                              self.unet.config.sample_size[0], self.unet.config.sample_size[1])
                
                generated_imgs = self.forward(img_shape=noise_shape,
                                             num_inference_steps=self.num_inference_steps
                )['samples']
                
                # Visualization: Generated samples
                gen_dict = {'gen': generated_imgs}

                # Log the generated samples to the experiment logger
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=gen_dict,
                    keys=list(gen_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),
                    prefix='val'
                )
            
        return {"val_loss": img_mse}

    def configure_optimizers(self):
        """Configure optimizer for the UNet model."""
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.learning_rate,
            betas=tuple(self.betas),
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['unet'] = self.unet.state_dict()
        if self.use_ema:
            checkpoint['unet_ema'] = self.unet_ema.state_dict()
     
    def on_load_checkpoint(self, checkpoint):
        if 'unet' in checkpoint:
            self.unet.load_state_dict(checkpoint['unet'])
        if self.use_ema and 'unet_ema' in checkpoint:
            self.unet_ema.load_state_dict(checkpoint['unet_ema'])
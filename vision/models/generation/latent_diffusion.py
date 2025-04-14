import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy

from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.unets import UNet2DModel
from diffusers import DDIMScheduler

from vision.visualizers.basic_visualizer import visualize_comparisons
from vision.dist_samplers.basic_sampler import inverse_add_noise, DefaultSampler

class LatentDiffusionModel(pl.LightningModule):
    """Latent Diffusion Model as a Lightning module.
    
    This model performs diffusion in the latent space without any conditioning.
    Unlike text-conditioned models, this model learns to generate samples
    directly from the learned data distribution without external guidance.
    """
    
    def __init__(self, model_config, optimizer_config, validation_config):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])
        
        # Load Pre-train VAE
        self.vae_pretrained = model_config.get('vae_pretrained', None)
        print(f"Loading pretrained VAE from {self.vae_pretrained}")
        self.vae = AutoencoderKL.from_pretrained(self.vae_pretrained)
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        # Create a Unet Model
        self.unet_config = model_config.get('unet_config', {})
        self.unet = UNet2DModel.from_config(self.unet_config)

        # Create Schedulers
        self.train_scheduler_config = model_config.get('train_scheduler_config', {})
        self.test_scheduler_config = model_config.get('test_scheduler_config', {})
        self.train_scheduler = DDIMScheduler.from_config(self.train_scheduler_config)
        self.test_scheduler = DDIMScheduler.from_config(self.test_scheduler_config)

        # 
        self.eta = model_config.get('eta', None)
        self.num_inference_steps = model_config.get('num_inference_steps', None)
        
        # Validation Setting (move to validation in the future)
        self.num_vis_samples = validation_config.get('num_vis_samples', 4)

        self.automatic_optimization = False

        # Training params
        self.learning_rate = optimizer_config.get('learning_rate', 1e-4)
        self.betas = optimizer_config.get('betas', [0.9, 0.999])
        self.weight_decay = optimizer_config.get('weight_decay', 0.0)
        self.use_ema = optimizer_config.get('use_ema', True)
        self.ema_decay = optimizer_config.get('ema_decay', 0.999)
        self.ema_start = optimizer_config.get('ema_start', 1000)
        self.noise_offset_weight = optimizer_config.get('noise_offset_weight', 0.0)
        
        self.test_sampler = DefaultSampler(self.unet, self.test_scheduler)
        
        # Create EMA model if requested
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
    
    def forward(self, batch_size=None, latent_shape=None, num_inference_steps=50, return_dict=True):
        """Generate new samples from random noise using latent diffusion.
        
        Args:
            batch_size (int, optional): Number of samples to generate (if latent_shape not provided)
            latent_shape (tuple, optional): Shape of latent noise (B, C, H, W); overrides batch_size
            num_inference_steps (int): Number of denoising steps
            return_dict (bool): Whether to return a dictionary with results
        """
        if latent_shape is None:
            if batch_size is None:
                raise ValueError("Must provide either batch_size or latent_shape")
            # Default latent shape based on VAE downsampling (e.g., H/8, W/8)
            latent_shape = (batch_size, 4, self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        
        # Generate random noise in latent space
        latents = torch.randn(latent_shape, device=self.device, dtype=self.unet.dtype)
        
        # Sample using the test sampler
        generated_latents = self.test_sampler.sample(
            latents,
            num_inference_steps=num_inference_steps,
            eta=self.eta,
            generator=None
        )
        
        # Decode to image space
        images = self.decode_latents(generated_latents)
        
        if return_dict:
            return {'samples': images, 'latents': generated_latents}
        else:
            return images

    def training_step(self, batch, batch_idx):
        """Training step for the Latent Diffusion Model."""
        # Extract images from batch
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        
        # Process the batch
        batch_size = real_imgs.shape[0]
        
        # Encode the images to latent space using frozen VAE
        with torch.no_grad():
            latents = self.vae.encode(real_imgs).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        
        # Apply noise offset for improved training if enabled (disabled)
        # if self.noise_offset_weight > 0:
        #     noise = noise + self.noise_offset_weight * torch.randn(
        #         latents.shape[0], latents.shape[1], 1, 1, device=noise.device)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to latents according to noise schedule
        noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)
        
        # Determine target based on prediction type
        if self.train_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.train_scheduler.config.prediction_type == "v_prediction":
            target = self.train_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.train_scheduler.config.prediction_type}")
        
        # Get the model prediction
        noise_pred = self.unet(noisy_latents, timesteps).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred.float(), target.float())
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        # Update EMA model after each step
        if self.use_ema and self.global_step >= self.ema_start:
            self._update_ema()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''
        
        '''
        self.test_scheduler.set_timesteps(self.test_scheduler.config.num_train_timesteps)

        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.size(0)
        
        # Generate sample images using the current model
        with torch.no_grad():

            # 1. Evaluate VAE reconstruction quality:
            # - Encode real images to get latent distribution
            # - Sample from this distribution to get latent representations
            latent_dist = self.vae.encode(real_imgs).latent_dist
            latents = latent_dist.sample()

            # 2. Sample random timesteps from diffusion schedule
            # This simulates different points in the diffusion process
            timesteps = torch.randint(
                0, self.test_scheduler.config.num_train_timesteps, 
                (batch_size,), device=self.device
            ).long()

            # 3. Generate random noise of the same shape as latents
            # This represents the noise to be predicted by the UNet
            noise = torch.randn_like(latents)

            # 4. Add noise to the latents according to the sampled timesteps
            # This creates noisy latents that the UNet will try to denoise
            noisy_latents = self.test_scheduler.add_noise(latents, noise, timesteps)

            # 5. Predict noise using the UNet model
            # The UNet takes noisy latents and timesteps as input
            noise_pred = self.unet(noisy_latents, timesteps).sample

            # 7. Calculate MSE between predicted noise and actual noise
            # This is the primary metric for diffusion model training
            latent_mse = F.mse_loss(noise_pred, noise)
            self.log("val/latent_mse", latent_mse)  # Core diffusion model metric

            # 8. Reconstruct the denoised latent by inverting the noise addition process
            # This applies the inverse diffusion process using predicted noise
            latent_denoised = inverse_add_noise(noisy_latents, noise_pred, timesteps, self.test_scheduler)

            # 9. Decode the denoised latents to get reconstructed images
            recon_imgs = self.vae.decode(latent_denoised).sample

            # 10. Calculate pixel-space MSE between reconstructed and real images
            # This provides an additional metric for image quality
            img_mse = F.mse_loss(recon_imgs, real_imgs)
            self.log("val/img_mse", img_mse)  # Core diffusion model metric

            # 11. For the first batch only, create and log visualization samples
            if batch_idx == 0:
                # 12. Generate new samples from random noise using the test sampler
                # This demonstrates the model's generative capabilities
                noise_shape = list(latent_dist.mean.shape)
                noise_shape[0] = self.num_vis_samples               # Set batch size to visualization sample count
                latents = self.test_sampler.sample(noise_shape, 
                                                   num_inference_steps=self.num_inference_steps, 
                                                   eta=self.eta,    # Controls stochasticity in DDIM sampling
                                                   )

                # 13. Decode the generated latents to get sample images
                samples = self.decode_latents(latents)

                # 14. Create comparison visualizations between real and reconstructed images
                # This helps assess reconstruction quality visually
                cmp_dict = {
                    'real': real_imgs[:self.num_vis_samples],
                    'recon': recon_imgs[:self.num_vis_samples]
                }

                # 15. Log the comparison visualizations to the experiment logger
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=cmp_dict,
                    keys=list(cmp_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),         # Weights and biases configuration
                    prefix='val'            # Prefix for logging
                )

                # 16. Create and log visualization of purely generated samples
                # This shows what the model can generate from scratch
                gen_dict = {
                    'generated': samples[:4],
                }

                # 17. Log the generated samples to the experiment logger
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=gen_dict,
                    keys=list(gen_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),
                    prefix='val'
                )                
        # 18. Return the primary validation metric (noise prediction error)
        return {"val_loss": latent_mse}
        
    def configure_optimizers(self):
        """Configure optimizer for the UNet model only (VAE is frozen)."""
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.learning_rate,
            betas=tuple(self.betas),
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def decode_latents(self, latents):
        """Decode latents to images using the VAE."""
        # Scale latents according to VAE configuration
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
            
        return image
    
    def on_save_checkpoint(self, checkpoint):
        """Save EMA model state."""
        if self.use_ema:
            checkpoint['unet_ema'] = self.unet_ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """Load EMA model state."""
        if self.use_ema and 'unet_ema' in checkpoint:
            self.unet_ema.load_state_dict(checkpoint['unet_ema'])
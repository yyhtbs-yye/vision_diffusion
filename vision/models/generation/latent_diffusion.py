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

    def __init__(self, model_config, train_config, validation_config):
        super().__init__()
        # !!! Disable automatic optimization
        self.automatic_optimization = False

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
        self.latent_channels = model_config.get('latent_channels', 4)
        self.unet_config['in_channels'] = model_config.get('latent_channels', 4)
        self.unet_config['out_channels'] = model_config.get('latent_channels', 4)
        self.unet = UNet2DModel.from_config(self.unet_config)

        # Create Schedulers
        self.train_scheduler_config = model_config.get('train_scheduler_config', {})
        self.valid_scheduler_config = model_config.get('test_scheduler_config', {})
        self.sample_scheduler_config = model_config.get('test_scheduler_config', {})
        self.train_scheduler = DDIMScheduler.from_config(self.train_scheduler_config)
        self.valid_scheduler = DDIMScheduler.from_config(self.valid_scheduler_config)
        self.sample_scheduler = DDIMScheduler.from_config(self.sample_scheduler_config)

        # Sampling parameters
        self.eta = validation_config.get('eta', None)
        self.num_inference_steps = validation_config.get('num_inference_steps', 50)
        self.num_vis_samples = validation_config.get('num_vis_samples', 4)

        optimizer_config = train_config.get('optimizer_config', {})
        self.learning_rate = optimizer_config.get('learning_rate', 1e-4)
        self.betas = optimizer_config.get('betas', [0.9, 0.999])
        self.weight_decay = optimizer_config.get('weight_decay', 0.0)
        self.use_ema = optimizer_config.get('use_ema', True)
        self.ema_decay = optimizer_config.get('ema_decay', 0.999)
        self.ema_start = optimizer_config.get('ema_start', 1000)
        self.noise_offset_weight = optimizer_config.get('noise_offset_weight', 0.0)
        
        self.sampler = DefaultSampler(self.unet, self.sample_scheduler)
        
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

        if latent_shape is None:
            if batch_size is None:
                raise ValueError("Must provide either batch_size or latent_shape")
            
            latent_shape = (batch_size, self.latent_channels, 
                            self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        
        # Generate random noise in latent space
        latents = torch.randn(latent_shape, device=self.device, dtype=self.unet.dtype)
        
        # Sample using the test sampler
        generated_latents = self.test_sampler.sample(latents, eta=self.eta,
                                                     num_inference_steps=num_inference_steps,
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
            latents = self.encode_latents(real_imgs)
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        
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

        # Manually optimize
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        # Update EMA model after each step
        if self.use_ema and self.global_step >= self.ema_start:
            self._update_ema()
        
        return loss
    
    def validation_step(self, batch, batch_idx):

        self.valid_scheduler.set_timesteps(self.valid_scheduler.config.num_train_timesteps)

        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.size(0)
        
        # Generate sample images using the current model
        with torch.no_grad():
            # Encode the images to latent space using frozen VAE
            latents = self.encode_latents(real_imgs)

            # Sample random timesteps
            timesteps = torch.randint(
                0, self.valid_scheduler.config.num_train_timesteps, 
                (batch_size,), device=self.device
            ).long()

            # Add noise to latents according to noise schedule
            noise = torch.randn_like(latents)

            # Generate random noise of the same shape as latents
            noisy_latents = self.valid_scheduler.add_noise(latents, noise, timesteps)

            # The UNet takes noisy latents and timesteps as input
            noise_pred = self.unet(noisy_latents, timesteps).sample

            # Calculate MSE between predicted noise and actual noise
            latent_mse = F.mse_loss(noise_pred, noise)
            self.log("val/latent_mse", latent_mse)  # Core diffusion model metric

            # 11. For the first batch only, create and log visualization samples
            if batch_idx == 0:
                
                # This applies the inverse diffusion process using predicted noise
                denoised_latent = inverse_add_noise(noisy_latents, noise_pred, timesteps, self.valid_scheduler)

                # Decode the denoised latents to get reconstructed images
                recon_imgs = self.decode_latents(denoised_latent)

                # This helps assess reconstruction quality visually
                cmp_dict = {
                    'real': real_imgs[:self.num_vis_samples],
                    'recon': recon_imgs[:self.num_vis_samples]
                }

                # Log the comparison visualizations to the experiment logger
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=cmp_dict,
                    keys=list(cmp_dict.keys()),
                    global_step=self.global_step,
                    wnb=(1.0, 0),         # Weights and biases configuration
                    prefix='val'            # Prefix for logging
                )

                noise_shape = [self.num_vis_samples, *latents.shape[1:]] 
                noise = torch.randn(noise_shape, device=self.device, dtype=self.unet.dtype)

                latents = self.sampler.sample(noise, num_inference_steps=self.num_inference_steps, eta=self.eta)

                # Decode the generated latents to get sample images
                generated_imgs = self.decode_latents(latents)

                # Create and log visualization of purely generated samples
                gen_dict = {
                    'gen': generated_imgs,
                }

                # Log the generated samples to the experiment logger
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=gen_dict,
                    keys=list(gen_dict.keys()),
                    global_step=self.global_step,
                    wnb=(1.0, 0),         # Weights and biases configuration
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
    def encode_latents(self, images):
        """Encode images to latents using the VAE."""
        # Encode images to latent space
        latents = self.vae.encode(images).latent_dist.mean
        latents = latents * self.vae.config.scaling_factor
        
        return latents
    
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['unet'] = self.unet.state_dict()  # Save main model
        if self.use_ema:
            checkpoint['unet_ema'] = self.unet_ema.state_dict()  # Save EMA model
     
    def on_load_checkpoint(self, checkpoint):
        # Load the main UNet model
        if 'unet' in checkpoint:
            self.unet.load_state_dict(checkpoint['unet'])
        # Load the EMA model if it exists and is used
        if self.use_ema and 'unet_ema' in checkpoint:
            self.unet_ema.load_state_dict(checkpoint['unet_ema'])
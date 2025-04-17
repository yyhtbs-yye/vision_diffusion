import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy

from diffusers.models.unets import UNet2DModel
from diffusers import DDIMScheduler

from vision.visualizers.basic_visualizer import visualize_comparisons
from vision.dist_samplers.basic_sampler import inverse_add_noise, DefaultSampler

class PixelDiffusionModel(pl.LightningModule):
    
    def __init__(self, model_config, train_config, validation_config):
        super().__init__()
        # Disable automatic optimization
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=['model_config'])
        
        # Create a UNet Model
        self.unet_config = model_config.get('unet_config', {})
        self.unet_config['in_channels'] = model_config.get('in_channels', 3)  # RGB images
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
        
        # Training params
        optimizer_config = train_config.get('optimizer', {})
        self.learning_rate = optimizer_config.get('learning_rate', 1e-4)
        self.betas = optimizer_config.get('betas', [0.9, 0.999])
        self.weight_decay = optimizer_config.get('weight_decay', 0.0)
        self.use_ema = optimizer_config.get('use_ema', True)
        self.ema_decay = optimizer_config.get('ema_decay', 0.999)
        self.ema_start = optimizer_config.get('ema_start', 1000)
        self.noise_offset_weight = optimizer_config.get('noise_offset_weight', 0.0)
        
        self.sampler = DefaultSampler(self.unet, self.sample_scheduler)
        
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
    
    def forward(self, batch_size=None, img_shape=None, num_inference_steps=50, return_dict=True):

        if img_shape is None:
            if batch_size is None:
                raise ValueError("Must provide either batch_size or img_shape")
            # Default image shape based on UNet configuration
            img_shape = (batch_size, self.unet_config['in_channels'], 
                         self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        
        # Generate random noise in image space
        noise = torch.randn(img_shape, device=self.device, dtype=self.unet.dtype)
        
        # Sample using the test sampler
        generated_images = self.test_sampler.sample(
            noise,
            num_inference_steps=num_inference_steps,
            eta=self.eta,
            generator=None
        )
        
        generated_images = generated_images * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        
        if return_dict:
            return {'samples': generated_images}
        else:
            return generated_images

    def training_step(self, batch, batch_idx):

        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.shape[0]
        
        # Sample random timesteps
        noise = torch.randn_like(real_imgs)
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to images according to noise schedule
        noisy_imgs = self.train_scheduler.add_noise(real_imgs, noise, timesteps)
        
        # Determine target based on prediction type
        if self.train_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.train_scheduler.config.prediction_type == "v_prediction":
            target = self.train_scheduler.get_velocity(real_imgs, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.train_scheduler.config.prediction_type}")
        
        # Get the model prediction
        noise_pred = self.unet(noisy_imgs, timesteps).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred.float(), target.float())
        
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

        self.valid_scheduler.set_timesteps(self.valid_scheduler.config.num_train_timesteps)

        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.size(0)
        
        # Generate sample images using the current model
        with torch.no_grad():
            # Sample random timesteps from diffusion schedule
            timesteps = torch.randint(
                0, self.valid_scheduler.config.num_train_timesteps, 
                (batch_size,), device=self.device
            ).long()

            # Generate random noise of the same shape as images
            noise = torch.randn_like(real_imgs)

            # Add noise to the images according to the sampled timesteps
            noisy_imgs = self.valid_scheduler.add_noise(real_imgs, noise, timesteps)

            # Predict noise using the UNet model
            noise_pred = self.unet(noisy_imgs, timesteps).sample

            # Calculate MSE between predicted noise and actual noise
            img_mse = F.mse_loss(noise_pred, noise)
            self.log("val/img_mse", img_mse)  # Core diffusion model metric

            # For the first batch only, create and log visualization samples
            if batch_idx == 0:

                # Reconstruct the denoised image by inverting the noise addition process
                img_denoised = inverse_add_noise(noisy_imgs, noise_pred, timesteps, self.valid_scheduler)

                # Create comparison visualizations between real and reconstructed images
                cmp_dict = {
                    'real': real_imgs[:self.num_vis_samples],
                    'recon': img_denoised[:self.num_vis_samples]
                }

                # Log the comparison visualizations to the experiment logger
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=cmp_dict,
                    keys=list(cmp_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),         # Weights and biases configuration
                    prefix='val'            # Prefix for logging
                )

                # Generate new samples from random noise using the test sampler
                noise_shape = (self.num_vis_samples, self.unet_config['in_channels'], 
                              self.unet.config.sample_size[0], self.unet.config.sample_size[1])
                
                noise = torch.randn(noise_shape, device=self.device, dtype=self.unet.dtype)

                generated_imgs = self.sampler.sample(noise, num_inference_steps=self.num_inference_steps, eta=self.eta)

                # Create and log visualization of purely generated samples
                gen_dict = {'gen': generated_imgs,}

                # Log the generated samples to the experiment logger
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=gen_dict,
                    keys=list(gen_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),
                    prefix='val'
                )                
        
        # Return the primary validation metric (noise prediction error)
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
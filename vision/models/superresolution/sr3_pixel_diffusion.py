import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy

from diffusers.models.unets import UNet2DModel
from diffusers import DDIMScheduler

from vision.visualizers.basic_visualizer import visualize_comparisons
from vision.dist_samplers.conditioned_basic_sampler import inverse_add_noise, DefaultSampler

class SR3PixelDiffusionModel(pl.LightningModule):
    
    def __init__(self, model_config, train_config, validation_config):
        super().__init__()
        # !!! Disable automatic optimization
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=['model_config'])
        
        # Create a Unet Model (adjusted to work with image dimensions)
        self.unet_config = model_config.get('unet_config', {})
        # Ensure the UNet is configured for pixel space (right channels and dimensions)
        self.unet_config['out_channels'] = model_config.get('in_channels', 3)
        self.unet_config['in_channels'] = model_config.get('in_channels', 3) * 2  # Support LR (c) HR
        self.unet = UNet2DModel.from_config(self.unet_config)
        # Different from Generation, scale is resolution factor
        self.scale = model_config.get('scale', 2)

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
    
    def forward(self, lqs, num_inference_steps=50, return_dict=True):

        batch_size, c, lq_h, lq_w = lqs.shape

        hq_h = lq_h * self.scale
        hq_w = lq_w * self.scale

        # Generate random noise in image space
        noise = torch.randn(batch_size, c, hq_h, hq_w, 
                            device=self.device, dtype=self.unet.dtype)
        
        # !!!!this can be improved using SRCNN and PixelShuffle
        uqs = F.interpolate(lqs, size=(hq_h, hq_w), 
                            mode='bilinear', align_corners=False)

        # Sample using the test sampler
        generated_images = self.sampler.sample(
            noise, uqs,
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

        hqs = batch['gt']

        batch_size, c, hq_h, hq_w = hqs.shape

        uqs = F.interpolate(batch['lq'], size=(hq_h, hq_w), 
                            mode='bilinear', align_corners=False)
        
        # Add noise to images
        noise = torch.randn_like(hqs)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to images according to noise schedule
        noisy_imgs = self.train_scheduler.add_noise(hqs, noise, timesteps)
        
        cnoisy_imgs = torch.cat((noisy_imgs, uqs), dim=1)

        # Determine target based on prediction type
        if self.train_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.train_scheduler.config.prediction_type == "v_prediction":
            target = self.train_scheduler.get_velocity(hqs, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.train_scheduler.config.prediction_type}")
        
        # Get the model prediction
        noise_pred = self.unet(cnoisy_imgs, timesteps).sample
        
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

        hqs = batch['gt']

        batch_size, c, hq_h, hq_w = hqs.shape
        
        
        # Generate sample images using the current model
        with torch.no_grad():

            uqs = F.interpolate(batch['lq'], size=(hq_h, hq_w), 
                                mode='bilinear', align_corners=False)

            noise = torch.randn_like(hqs)

            # Sample random timesteps from diffusion schedule
            timesteps = torch.randint(
                0, self.valid_scheduler.config.num_train_timesteps, 
                (batch_size,), device=self.device
            ).long()

            # Generate random noise of the same shape as images
            noisy_imgs = self.valid_scheduler.add_noise(hqs, noise, timesteps)
            
            cnoisy_imgs = torch.cat((noisy_imgs, uqs), dim=1)

            # Predict noise using the UNet model
            noise_pred = self.unet(cnoisy_imgs, timesteps).sample

            # Calculate MSE between predicted noise and actual noise
            img_mse = F.mse_loss(noise_pred, noise)
            self.log("val/img_mse", img_mse)  # Core diffusion model metric

            # For the first batch only, create and log visualization samples
            if batch_idx == 0:

                # Reconstruct the denoised image by inverting the noise addition process
                img_denoised = inverse_add_noise(noisy_imgs, noise_pred, timesteps, self.valid_scheduler)

                # Create comparison visualizations between real and reconstructed images
                cmp_dict = {
                    'real': hqs[:self.num_vis_samples],
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

                noise = torch.randn_like(uqs)
                
                sr_imgs = self.sampler.sample(noise, uqs, 
                                              num_inference_steps=self.num_inference_steps, 
                                              eta=self.eta)

                # Create and log visualization of purely generated samples
                gen_dict = {
                    'superresolution': sr_imgs,
                }

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
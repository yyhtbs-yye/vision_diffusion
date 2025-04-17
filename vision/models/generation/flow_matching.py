import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy

from diffusers.models.unets import UNet2DModel

from vision.visualizers.basic_visualizer import visualize_comparisons

class PixelFlowMatchingModel(pl.LightningModule):
    
    def __init__(self, model_config, train_config, validation_config):
        super().__init__()
        # Disable automatic optimization
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=['model_config'])
        
        # Create a UNet Model
        self.unet_config = model_config.get('unet_config', {})
        self.unet_config['in_channels'] = model_config.get('in_channels', 3)  # RGB images
        self.unet = UNet2DModel.from_config(self.unet_config)

        # Define number of timesteps for UNet compatibility (no scheduler needed)
        self.num_timesteps = 1000  # Matches typical DDPM setup

        # Sampling parameters
        self.eta = validation_config.get('eta', None)  # Not used in flow matching, kept for config compatibility
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
    
    def _integrate_ode(self, x, t_start, t_end, num_steps):
        """Integrate the ODE from t_start to t_end using Euler method."""
        x = x.clone()
        dt = (t_end - t_start) / num_steps
        for step in range(num_steps):
            t = t_start + step * (t_end - t_start) / num_steps
            timestep = int(t * (self.num_timesteps - 1))
            v_pred = self.unet(x, timestep).sample
            x = x + v_pred * dt
        return x

    def forward(self, batch_size=None, img_shape=None, num_inference_steps=50, return_dict=True):
        """Generate samples by integrating from t=1 (noise) to t=0 (data)."""
        if img_shape is None:
            if batch_size is None:
                raise ValueError("Must provide either batch_size or img_shape")
            img_shape = (batch_size, self.unet_config['in_channels'], 
                         self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        
        x = torch.randn(img_shape, device=self.device, dtype=self.unet.dtype)
        generated_images = self._integrate_ode(x, t_start=1.0, t_end=0.0, num_steps=num_inference_steps)
        generated_images = generated_images
        
        if return_dict:
            return {'samples': generated_images}
        else:
            return generated_images

    def training_step(self, batch, batch_idx):
        """Train the model to predict the velocity field using a linear OT path."""
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.num_timesteps, 
            (batch_size,), device=self.device
        ).long()
        t = timesteps / (self.num_timesteps - 1)  # t in [0,1], 0=data, 1=noise
        
        # Define path: x1=data, x0=noise
        x1 = real_imgs
        x0 = torch.randn_like(real_imgs)
        
        # Compute z_t along the linear path
        z_t = (1 - t.view(-1, 1, 1, 1)) * x1 + t.view(-1, 1, 1, 1) * x0
        
        # Predict velocity
        v_pred = self.unet(z_t, timesteps).sample
        
        # Target velocity from OT linear path
        v_target = x0 - x1
        
        # Loss: MSE between predicted and target velocity
        loss = F.mse_loss(v_pred, v_target)
        
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
        """Validate with single-step reconstruction, generation, and MSE metrics."""
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.size(0)

        with torch.no_grad():
            # Sample random timesteps
            timesteps = torch.randint(
                0, self.num_timesteps, 
                (real_imgs.size(0),), device=self.device
            ).long()

            t = timesteps / (self.num_timesteps - 1)
            # Perturb: z_t = (1-t) * x1 + t * x0
            x1 = real_imgs
            x0 = torch.randn_like(x1)
            z_t = (1 - t.view(-1, 1, 1, 1)) * x1 + t.view(-1, 1, 1, 1) * x0
            # Predict velocity and take one step back
            v_pred = self.unet(z_t, timesteps).sample
            
            dt = -t.view(-1, 1, 1, 1) 
            recon_imgs = z_t + v_pred * dt

            # Calculate MSE between predicted noise and actual noise
            img_mse = F.mse_loss(real_imgs, recon_imgs)
            self.log("val/img_mse", img_mse)  # Core diffusion model metric

            if batch_idx == 0:

                # Visualization: Real vs. Generated vs. Reconstructed
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
            
        return {"val_loss": img_mse}  # Use recon_mse as primary validation metric

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
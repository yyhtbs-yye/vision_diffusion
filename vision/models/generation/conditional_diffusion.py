import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy

from diffusers.models.unets import UNet2DConditionModel
from diffusers import DDIMScheduler

from vision.visualizers.basic_visualizer import visualize_comparisons
from vision.dist_samplers.basic_sampler import inverse_add_noise, DefaultSampler

class ConditionalEmbedding(nn.Module):
    """Embedding layer for conditional attributes."""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x):
        return self.embedding(x)

class ConditionalPixelDiffusionModel(pl.LightningModule):
    
    def __init__(self, model_config, train_config, validation_config):
        super().__init__()
        # Disable automatic optimization
        self.automatic_optimization = False

        self.save_hyperparameters(ignore=['model_config'])
        
        # Model configuration
        self.in_channels = model_config.get('in_channels', 3)  # RGB images
        
        # Conditional embedding config
        self.condition_config = model_config.get('condition_config', {})
        self.condition_dim = self.condition_config.get('condition_dim', 128)
        self.condition_feature_dim = self.condition_config.get('condition_feature_dim', 64)
        self.use_conditions = self.condition_config.get('use_conditions', True)
        
        # Create a UNet2DConditionModel for conditional generation
        self.unet_config = model_config.get('unet_config', {})
        self.unet_config.update({
            'in_channels': self.in_channels,
            'cross_attention_dim': self.condition_dim if self.use_conditions else None,
        })
        
        # Initialize the conditional UNet model
        self.unet = UNet2DConditionModel(**self.unet_config)
        
        # Conditional embedding network
        if self.use_conditions:
            self.condition_embedding = ConditionalEmbedding(
                input_dim=self.condition_feature_dim,
                embed_dim=self.condition_dim
            )

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
        
        # Create custom sampler for conditional generation
        self.sampler = self._create_conditional_sampler()
        
        # EMA model setup
        if self.use_ema:
            self.unet_ema = deepcopy(self.unet)
            if self.use_conditions:
                self.condition_embedding_ema = deepcopy(self.condition_embedding)
            for param in self.unet_ema.parameters():
                param.requires_grad = False
            if self.use_conditions:
                for param in self.condition_embedding_ema.parameters():
                    param.requires_grad = False
    
    def _create_conditional_sampler(self):
        """Create a sampler that supports conditional generation."""
        
        class ConditionalSampler(DefaultSampler):
            def __init__(self, unet, scheduler, embedding_model=None):
                super().__init__(unet, scheduler)
                self.embedding_model = embedding_model
                
            def sample(self, noise, num_inference_steps=50, eta=0.0, generator=None, condition=None):
                """Sample images from noise with optional conditioning."""
                # Set the scheduler timesteps
                self.scheduler.set_timesteps(num_inference_steps)
                
                # Initialize sample
                sample = noise
                
                # Process condition if provided
                cond_embed = None
                if condition is not None and self.embedding_model is not None:
                    cond_embed = self.embedding_model(condition)
                
                # Denoising loop
                for t in self.scheduler.timesteps:
                    # Expand the sample for input to the network
                    model_input = torch.cat([sample] * 2) if self.scheduler.config.prediction_type == "v_prediction" else sample
                    
                    # Predict the noise residual
                    with torch.no_grad():
                        # Pass condition through cross-attention
                        noise_pred = self.unet(
                            model_input, 
                            t,
                            encoder_hidden_states=cond_embed
                        ).sample
                    
                    # Perform guidance and compute the previous sample
                    sample = self.scheduler.step(noise_pred, t, sample, eta=eta, generator=generator).prev_sample
                
                return sample
        
        # Create and return the conditional sampler
        if self.use_conditions:
            return ConditionalSampler(self.unet, self.sample_scheduler, self.condition_embedding)
        else:
            return DefaultSampler(self.unet, self.sample_scheduler)
        
    def _update_ema(self):
        """Update EMA model parameters."""
        if not self.use_ema: return
        
        for ema_param, param in zip(self.unet_ema.parameters(), self.unet.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        
        # Update buffers
        for ema_buffer, buffer in zip(self.unet_ema.buffers(), self.unet.buffers()):
            ema_buffer.data.copy_(buffer.data)
            
        # Update condition embedding if used
        if self.use_conditions:
            for ema_param, param in zip(self.condition_embedding_ema.parameters(), self.condition_embedding.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
                
            for ema_buffer, buffer in zip(self.condition_embedding_ema.buffers(), self.condition_embedding.buffers()):
                ema_buffer.data.copy_(buffer.data)
    
    def _flatten_attributes(self, attributes):
        """Flatten attribute dictionary to a single tensor."""
        if isinstance(attributes, dict):
            # Concatenate all attribute tensors
            attr_tensors = []
            for key in sorted(attributes.keys()):  # Sort keys for consistency
                val = attributes[key]
                if isinstance(val, torch.Tensor):
                    # Ensure it's a float tensor and flatten it
                    attr_tensors.append(val.float().view(val.size(0), -1))
                    
            # Concatenate along feature dimension
            if attr_tensors:
                return torch.cat(attr_tensors, dim=1)
        
        # If attributes is already a tensor or no dict attributes found
        elif isinstance(attributes, torch.Tensor):
            return attributes.float()
            
        # Return None if no valid attributes
        return None
    
    def forward(self, batch_size=None, img_shape=None, num_inference_steps=50, condition=None, return_dict=True):
        """Generate samples with optional conditioning."""
        if img_shape is None:
            if batch_size is None:
                raise ValueError("Must provide either batch_size or img_shape")
            # Default image shape
            img_shape = (batch_size, self.in_channels, 
                         self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        
        # Generate random noise
        noise = torch.randn(img_shape, device=self.device, dtype=self.unet.dtype)
        
        # Sample using the conditional sampler
        generated_images = self.sampler.sample(
            noise,
            num_inference_steps=num_inference_steps,
            eta=self.eta,
            generator=None,
            condition=condition
        )
        
        # Scale to [0, 1] range
        generated_images = generated_images * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        
        if return_dict:
            return {'samples': generated_images}
        else:
            return generated_images

    def training_step(self, batch, batch_idx):
        """Training step with conditional generation."""
        # Extract images and conditional attributes
        if isinstance(batch, dict):
            real_imgs = batch.get('gt', batch.get('image'))
            attributes = batch.get('attributes', batch.get('attributes_combined'))
        else:
            real_imgs = batch
            attributes = None
            
        batch_size = real_imgs.shape[0]
        
        # Flatten attributes if necessary
        if attributes is not None and self.use_conditions:
            attributes = self._flatten_attributes(attributes)
        
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
        
        # Get the model prediction with conditional embedding if available
        if attributes is not None and self.use_conditions:
            # Embed the attributes
            condition_embed = self.condition_embedding(attributes)
            
            # Pass to UNet with cross-attention
            noise_pred = self.unet(
                noisy_imgs, 
                timesteps, 
                encoder_hidden_states=condition_embed
            ).sample
        else:
            # Unconditional case
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
        """Validation step with conditional generation."""
        self.valid_scheduler.set_timesteps(self.valid_scheduler.config.num_train_timesteps)

        # Extract images and conditional attributes
        if isinstance(batch, dict):
            real_imgs = batch.get('gt', batch.get('image'))
            attributes = batch.get('attributes', batch.get('attributes_combined'))
        else:
            real_imgs = batch
            attributes = None
            
        batch_size = real_imgs.size(0)
        
        # Flatten attributes if necessary
        if attributes is not None and self.use_conditions:
            attributes = self._flatten_attributes(attributes)
        
        # Generate sample images using the current model
        with torch.no_grad():
            # Sample random timesteps from diffusion schedule
            timesteps = torch.randint(
                0, self.valid_scheduler.config.num_train_timesteps, 
                (batch_size,), device=self.device
            ).long()

            # Generate random noise
            noise = torch.randn_like(real_imgs)

            # Add noise to the images
            noisy_imgs = self.valid_scheduler.add_noise(real_imgs, noise, timesteps)

            # Predict noise with conditions if available
            if attributes is not None and self.use_conditions:
                # Embed the attributes
                condition_embed = self.condition_embedding(attributes)
                
                # Pass to UNet with cross-attention
                noise_pred = self.unet(
                    noisy_imgs,
                    timesteps,
                    encoder_hidden_states=condition_embed
                ).sample
            else:
                # Unconditional case
                noise_pred = self.unet(noisy_imgs, timesteps).sample

            # Calculate MSE between predicted noise and actual noise
            img_mse = F.mse_loss(noise_pred, noise)
            self.log("val/img_mse", img_mse)

            # For the first batch only, create and log visualization samples
            if batch_idx == 0:
                # Reconstruct the denoised image
                img_denoised = inverse_add_noise(noisy_imgs, noise_pred, timesteps, self.valid_scheduler)

                # Create comparison visualizations
                cmp_dict = {
                    'real': real_imgs[:self.num_vis_samples],
                    'recon': img_denoised[:self.num_vis_samples]
                }

                # Log the comparison visualizations
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=cmp_dict,
                    keys=list(cmp_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),
                    prefix='val'
                )

                # Generate new samples from random noise with conditions
                noise_shape = (self.num_vis_samples, self.in_channels, 
                              self.unet.config.sample_size[0], self.unet.config.sample_size[1])
                
                noise = torch.randn(noise_shape, device=self.device, dtype=self.unet.dtype)
                
                # Use the first few attributes for conditioning if available
                gen_conditions = None
                if attributes is not None and self.use_conditions:
                    gen_conditions = attributes[:self.num_vis_samples]
                
                # Generate samples with conditions
                generated_imgs = self.sampler.sample(
                    noise, 
                    num_inference_steps=self.num_inference_steps, 
                    eta=self.eta,
                    condition=gen_conditions
                )

                # Create and log visualization of generated samples
                gen_dict = {'gen': generated_imgs}

                # Log the generated samples
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=gen_dict,
                    keys=list(gen_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),
                    prefix='val'
                )
        
        # Return the primary validation metric
        return {"val_loss": img_mse}
        
    def configure_optimizers(self):
        """Configure optimizer for all trainable parameters."""
        # Collect all parameters to optimize
        params_to_optimize = list(self.unet.parameters())
        
        # Add conditional embedding parameters if used
        if self.use_conditions:
            params_to_optimize.extend(list(self.condition_embedding.parameters()))
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.learning_rate,
            betas=tuple(self.betas),
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        """Save model state in checkpoint."""
        checkpoint['unet'] = self.unet.state_dict()
        if self.use_conditions:
            checkpoint['condition_embedding'] = self.condition_embedding.state_dict()
        if self.use_ema:
            checkpoint['unet_ema'] = self.unet_ema.state_dict()
            if self.use_conditions:
                checkpoint['condition_embedding_ema'] = self.condition_embedding_ema.state_dict()
     
    def on_load_checkpoint(self, checkpoint):
        """Load model state from checkpoint."""
        if 'unet' in checkpoint:
            self.unet.load_state_dict(checkpoint['unet'])
        if self.use_conditions and 'condition_embedding' in checkpoint:
            self.condition_embedding.load_state_dict(checkpoint['condition_embedding'])
        if self.use_ema and 'unet_ema' in checkpoint:
            self.unet_ema.load_state_dict(checkpoint['unet_ema'])
        if self.use_ema and self.use_conditions and 'condition_embedding_ema' in checkpoint:
            self.condition_embedding_ema.load_state_dict(checkpoint['condition_embedding_ema'])
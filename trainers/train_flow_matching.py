import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy

from diffusers.models.unets import UNet2DModel

from vision.visualizers.basic_visualizer import visualize_comparisons
from vision.solvers.ode_solvers import FlowMatchingSolver

class FlowMatchingModel(pl.LightningModule):
    
    def __init__(self, model_config, optimizer_config, validation_config):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])
        
        # Create a Unet Model (for vector field prediction)
        self.unet_config = model_config.get('unet_config', {})
        # Ensure the UNet is configured for pixel space
        self.unet_config['in_channels'] = model_config.get('in_channels', 3)  # RGB images
        self.unet = UNet2DModel.from_config(self.unet_config)

        # Sampling parameters
        self.num_inference_steps = model_config.get('num_inference_steps', 50)
        
        # Validation Setting
        self.num_vis_samples = validation_config.get('num_vis_samples', 4)

        self.automatic_optimization = False

        # Training params
        self.learning_rate = optimizer_config.get('learning_rate', 1e-4)
        self.betas = optimizer_config.get('betas', [0.9, 0.999])
        self.weight_decay = optimizer_config.get('weight_decay', 0.0)
        self.use_ema = optimizer_config.get('use_ema', True)
        self.ema_decay = optimizer_config.get('ema_decay', 0.999)
        self.ema_start = optimizer_config.get('ema_start', 1000)
        
        # Flow matching specific parameters
        self.sigma = model_config.get('sigma', 0.1)  # Noise level for SDE
        self.flow_type = model_config.get('flow_type', 'rectified_flow')
        self.solver_type = model_config.get('solver_type', 'rk4')
        self.rtol = model_config.get('rtol', 1e-3)
        self.atol = model_config.get('atol', 1e-3)
        
        # Normalize images to [-1, 1] range
        self.normalize_input = model_config.get('normalize_input', True)
        
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
    
    def forward(self, batch_size=None, img_shape=None, num_inference_steps=None, 
                solver_type=None, return_trajectory=False, return_dict=True):
        """Generate new samples using Flow Matching.
        
        Args:
            batch_size (int, optional): Number of samples to generate (if img_shape not provided)
            img_shape (tuple, optional): Shape of image noise (B, C, H, W); overrides batch_size
            num_inference_steps (int, optional): Number of integration steps
            solver_type (str, optional): ODE solver type to use
            return_trajectory (bool): Whether to return the integration trajectory
            return_dict (bool): Whether to return a dictionary with results
        """
        if img_shape is None:
            if batch_size is None:
                raise ValueError("Must provide either batch_size or img_shape")
            # Default image shape based on UNet configuration
            img_shape = (batch_size, self.unet_config['in_channels'], 
                         self.unet.config.sample_size[0], self.unet.config.sample_size[1])
        
        # Generate random noise as starting point
        noise = torch.randn(img_shape, device=self.device, dtype=self.unet.dtype)
        
        # Use the active model (EMA if available and past start point)
        unet = self.unet_ema if self.use_ema and self.global_step >= self.ema_start else self.unet
        
        # Create solver with current parameters
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        solver_type = solver_type if solver_type is not None else self.solver_type
        
        solver = FlowMatchingSolver(
            model=unet,
            flow_type=self.flow_type,
            solver_type=solver_type,
            steps=steps,
            rtol=self.rtol,
            atol=self.atol,
            device=self.device
        )
        
        # Generate samples
        result = solver.sample(
            noise=noise,
            return_trajectory=return_trajectory
        )
        
        # Process result based on whether trajectory was requested
        if return_trajectory:
            generated_images, trajectory = result
        else:
            generated_images = result
            trajectory = None
        
        # Denormalize if needed
        if self.normalize_input:
            generated_images = (generated_images + 1) / 2  # [-1, 1] -> [0, 1]
            if trajectory is not None:
                trajectory = [(t + 1) / 2 for t in trajectory]
        
        if return_dict:
            result_dict = {'samples': generated_images}
            if trajectory is not None:
                result_dict['trajectory'] = trajectory
            return result_dict
        else:
            return generated_images if trajectory is None else (generated_images, trajectory)

    def _normalize_images(self, images):
        """Normalize images to [-1, 1] range if needed."""
        if self.normalize_input:
            return 2 * images - 1  # [0, 1] -> [-1, 1]
        return images
    
    def _denormalize_images(self, images):
        """Denormalize images from [-1, 1] to [0, 1] range if needed."""
        if self.normalize_input:
            return (images + 1) / 2  # [-1, 1] -> [0, 1]
        return images
    
    def conditional_vector_field(self, x_t, t, x_0, x_1):
        """Compute the conditional vector field for flow matching.
        
        Args:
            x_t: Current state at time t
            t: Time parameter in [0, 1]
            x_0: Initial state (noise)
            x_1: Target state (real image)
            
        Returns:
            Vector field v(x, t)
        """
        if self.flow_type == 'rectified_flow':
            # For rectified flow, the vector field is simply the direction from x_0 to x_1
            return x_1 - x_0
        else:  # VP-SDE or probability flow
            # For probability flow, the vector field depends on the score function
            # We approximate this by interpolating between noise and data
            sigma_t = self.sigma * t  # Time-dependent noise level
            drift = (x_1 - x_t) / (1 - t + 1e-5)  # Drift term (approximation)
            return drift

    def training_step(self, batch, batch_idx):
        """Training step for the Flow Matching Model."""
        # Extract images from batch
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        
        # Process the batch
        batch_size = real_imgs.shape[0]
        
        # Normalize images to [-1, 1] range if needed
        real_imgs = self._normalize_images(real_imgs)
        
        # Sample time uniformly between 0 and 1
        t = torch.rand(batch_size, device=self.device)
        
        # Sample noise for initial distribution - unrelated to specific real images
        z_0 = torch.randn_like(real_imgs)
        
        # Sample from the marginal distribution at time t by interpolating
        # This creates a valid sample from the path distribution without assuming
        # deterministic mappings between specific noise and image pairs
        t_expanded = t.view(-1, 1, 1, 1)
        
        # Path interpolation - for each batch element, interpolate between noise
        # and real image independently (not creating deterministic pairings)
        x_t = (1 - t_expanded) * z_0 + t_expanded * real_imgs
        
        # Add small noise to make the path stochastic if needed
        if self.sigma > 0:
            noise = torch.randn_like(x_t) * self.sigma * torch.sqrt(t_expanded * (1 - t_expanded))
            x_t = x_t + noise
        
        # Calculate the conditional vector field (ground truth flow)
        target_vector_field = self.conditional_vector_field(x_t, t_expanded, z_0, real_imgs)
        
        # Get model's predicted vector field
        pred_vector_field = self.unet(x_t, t).sample
        
        # Calculate flow matching loss
        # The loss minimizes the distance between the predicted and ideal vector fields
        loss = F.mse_loss(pred_vector_field.float(), target_vector_field.float())
        
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
        """Validation step for the Flow Matching Model."""
        real_imgs = batch['gt'] if isinstance(batch, dict) else batch
        batch_size = real_imgs.size(0)
        
        # Normalize images to [-1, 1] range if needed
        real_imgs_norm = self._normalize_images(real_imgs)
        
        # Generate sample images using the current model
        with torch.no_grad():
            # Sample time uniformly between 0 and 1
            t = torch.rand(batch_size, device=self.device)
            t_expanded = t.view(-1, 1, 1, 1)
            
            # Sample noise for initial distribution
            z_0 = torch.randn_like(real_imgs_norm)
            
            # Sample from the marginal distribution at time t
            x_t = (1 - t_expanded) * z_0 + t_expanded * real_imgs_norm
            
            # Add small noise to make the path stochastic if needed
            if self.sigma > 0:
                noise = torch.randn_like(x_t) * self.sigma * torch.sqrt(t_expanded * (1 - t_expanded))
                x_t = x_t + noise
            
            # Calculate the conditional vector field (ground truth flow)
            target_vector_field = self.conditional_vector_field(x_t, t_expanded, z_0, real_imgs_norm)
            
            # Get model's predicted vector field
            pred_vector_field = self.unet(x_t, t).sample
            
            # Calculate validation loss
            flow_mse = F.mse_loss(pred_vector_field, target_vector_field)
            self.log("val/flow_mse", flow_mse)
            
            # For the first batch only, create and log visualization samples
            if batch_idx == 0:
                # Use the active model (EMA if available and past start point)
                unet = self.unet_ema if self.use_ema and self.global_step >= self.ema_start else self.unet
                
                # Create solver with current parameters
                solver = FlowMatchingSolver(
                    model=unet,
                    flow_type=self.flow_type,
                    solver_type=self.solver_type,
                    steps=self.num_inference_steps,
                    rtol=self.rtol,
                    atol=self.atol,
                    device=self.device
                )
                
                # Generate samples
                noise_shape = (self.num_vis_samples, self.unet_config['in_channels'], 
                              self.unet.config.sample_size[0], self.unet.config.sample_size[1])
                
                noise = torch.randn(noise_shape, device=self.device, dtype=self.unet.dtype)
                generated_imgs = solver.sample(noise)
                
                # Denormalize generated images if needed
                generated_imgs_vis = self._denormalize_images(generated_imgs)
                
                # Create comparison visualizations between real and generated images
                cmp_dict = {
                    'real': real_imgs[:self.num_vis_samples],
                    'generated': generated_imgs_vis
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
                
                # Visualize a trajectory (less frequently)
                if self.global_step % 1000 == 0:
                    # Generate a trajectory from a single noise sample
                    single_noise = torch.randn(
                        (1, self.unet_config['in_channels'], 
                         self.unet.config.sample_size[0], self.unet.config.sample_size[1]),
                        device=self.device, dtype=self.unet.dtype
                    )
                    
                    _, trajectory = solver.sample(
                        noise=single_noise,
                        return_trajectory=True
                    )
                    
                    # Select a subset of frames from the trajectory
                    num_frames = min(8, len(trajectory))
                    indices = torch.linspace(0, len(trajectory)-1, num_frames).long()
                    trajectory_subset = [trajectory[i.item()] for i in indices]
                    
                    # Denormalize the trajectory frames
                    trajectory_subset = [self._denormalize_images(t) for t in trajectory_subset]
                    
                    # Create visualization dictionary
                    traj_dict = {f'frame_{i}': frame for i, frame in enumerate(trajectory_subset)}
                    
                    # Log the trajectory visualizations
                    visualize_comparisons(
                        logger=self.logger.experiment,
                        images_dict=traj_dict,
                        keys=list(traj_dict.keys()),
                        global_step=self.global_step,
                        wnb=(0.5, 0.5),
                        prefix='trajectory'
                    )
        
        # Return the validation loss
        return {"val_loss": flow_mse}
        
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
        """Save EMA model state."""
        if self.use_ema:
            checkpoint['unet_ema'] = self.unet_ema.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """Load EMA model state."""
        if self.use_ema and 'unet_ema' in checkpoint:
            self.unet_ema.load_state_dict(checkpoint['unet_ema'])
import torch

class FlowSampler:
    """Sampler for Flow Matching models.
    
    This sampler implements numerical integration methods to solve the ODE 
    defined by the learned vector field.
    """
    
    def __init__(self, unet, steps=50, flow_type="rectified_flow", 
                 solver="euler", device=None):
        """
        Args:
            unet: UNet model that predicts the vector field
            steps: Number of integration steps
            flow_type: Type of flow - 'rectified_flow' or 'probability_flow'
            solver: Numerical solver - 'euler' or 'heun'
            device: Device to perform computation on
        """
        self.unet = unet
        self.steps = steps
        self.flow_type = flow_type
        self.solver = solver
        self.device = device
        
    def sample(self, noise, unet=None, steps=None, return_trajectory=False):
        """Generate samples by integrating the flow starting from random noise.
        
        Args:
            noise: Initial noise tensor (B, C, H, W)
            unet: UNet model to use (if None, use self.unet)
            steps: Number of integration steps (if None, use self.steps)
            return_trajectory: Whether to return the entire trajectory
            
        Returns:
            Generated samples, shape (B, C, H, W)
        """
        # Use provided UNet or default
        unet = unet if unet is not None else self.unet
        # Use provided steps or default
        steps = steps if steps is not None else self.steps
        
        # Set device
        device = self.device if self.device is not None else noise.device
        
        # Initial state (random noise)
        x = noise.to(device)
        
        # Integration timesteps (from 0 to 1)
        # For flow matching, we integrate from t=0 (noise) to t=1 (data distribution)
        ts = torch.linspace(0, 1, steps + 1, device=device)
        dt = 1.0 / steps
        
        # Store trajectory if requested
        trajectory = [x.detach().cpu()] if return_trajectory else None
        
        # Numerical integration loop
        with torch.no_grad():
            for i in range(steps):
                # Current timestep
                t = ts[i] * torch.ones(x.shape[0], device=device)
                
                if self.solver == "euler":
                    # Euler method
                    v = unet(x, t).sample
                    x = x + dt * v
                elif self.solver == "heun":
                    # Heun's method (2nd order Runge-Kutta)
                    v1 = unet(x, t).sample
                    x_prime = x + dt * v1
                    
                    t_next = ts[i+1] * torch.ones(x.shape[0], device=device)
                    v2 = unet(x_prime, t_next).sample
                    
                    x = x + 0.5 * dt * (v1 + v2)
                else:
                    raise ValueError(f"Unknown solver: {self.solver}")
                
                # Store point in trajectory if requested
                if return_trajectory:
                    trajectory.append(x.detach().cpu())
        
        # Return the final samples and trajectory if requested
        if return_trajectory:
            return x, trajectory
        else:
            return x

def inverse_add_noise(noisy_images, noise_pred, timesteps, noise_scheduler):
    """Performs the inverse of the noise addition process for validation.
    This is not part of Flow Matching but included for compatibility with 
    diffusion-based validation pipelines.
    """
    # This ensures the function exists for compatibility, though in 
    # Flow Matching we typically don't use this approach
    alpha = noise_scheduler.alphas_cumprod[timesteps]
    alpha = alpha.view(-1, 1, 1, 1)
    
    # Predict x_0 from the predicted noise
    pred_original_image = (noisy_images - noise_pred * (1 - alpha).sqrt()) / alpha.sqrt()
    
    return pred_original_image
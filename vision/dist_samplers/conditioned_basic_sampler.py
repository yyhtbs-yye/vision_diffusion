import torch
from typing import Optional, Any
import inspect

def inverse_add_noise(noisy_latents, noise, timesteps, scheduler):
    # Get the noise coefficients used in the forward process
    alphas = scheduler.alphas_cumprod[timesteps]
    sigmas = (1 - alphas).sqrt()
    
    # Apply the inverse operation
    clean_latents = (noisy_latents - noise * sigmas.view(-1, 1, 1, 1)) / alphas.sqrt().view(-1, 1, 1, 1)
    return clean_latents


class DefaultSampler:
    """Default Simple diffusion sampler for generating output from diffusion models.
    
    This class handles the sampling process for diffusion models, providing an easy
    way to generate images without tightly coupling to any specific model architecture.
    """
    
    def __init__(self, tbone, scheduler):
        """Initialize a diffusion sampler.
        
        Args:
            tbone: Backbone model that takes (latents, timestep) and returns noise prediction
            scheduler: Diffusion scheduler that defines the diffusion process
        """
        self.tbone = tbone
        self.scheduler = scheduler
    
    @torch.no_grad()
    def sample(
        self,
        latents,
        condition,
        seed: Optional[int] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0,                   # DDIM eta parameter.
    ) -> torch.Tensor:

        # Initialize generator with seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.tbone.device).manual_seed(seed)
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Prepare latents
        latents = latents * self.scheduler.init_noise_sigma
        
        # Denoising loop
        timesteps_iter = self.scheduler.timesteps
        for i, t in enumerate(timesteps_iter):
            # Scale input according to timestep
            model_input = self.scheduler.scale_model_input(latents, t)

            model_input = torch.cat((model_input, condition), dim=1)
            
            # Predict noise
            noise_pred = self.tbone(model_input, t)

            if hasattr(noise_pred, 'sample'):
                noise_pred = noise_pred.sample
            
            # Update sample
            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]
       
        return latents
    
    def prepare_extra_step_kwargs(self, generator, eta):
        """Prepare extra kwargs for the scheduler step."""
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        
        # Check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs
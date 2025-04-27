import torch
import torch.nn.functional as F
from .ddpm import BaseDiffusionModel
from vision.solvers.sde_solvers import langevin_dynamics

class PixelScoreMatchingModel(BaseDiffusionModel):
    """Minimal score matching model with shared logic in base class."""

    def __init__(self, model_config, train_config, validation_config):
        super().__init__(model_config, train_config, validation_config)
        self.sigma_min = train_config.get('sigma_min', 0.01)
        self.sigma_max = train_config.get('sigma_max', 1.0)
        self.solver = langevin_dynamics
        self.solver_kwargs = {'sigma_min': self.sigma_min, 'sigma_max': self.sigma_max}
        self.log_sigma_min = torch.tensor(self.sigma_min).log()
        self.log_sigma_max = torch.tensor(self.sigma_max).log()

    def _map_to_noise_levels(self, t):
        log_sigma = self.log_sigma_min.to(t.device) + t * (self.log_sigma_max.to(t.device) - self.log_sigma_min.to(t.device))
        return torch.exp(log_sigma)

    def get_timesteps_from_time(self, t):
        sigmas = self._map_to_noise_levels(t)
        log_sigma = torch.log(sigmas)
        t_normalized = (log_sigma - self.log_sigma_min.to(t.device)) / (self.log_sigma_max.to(t.device) - self.log_sigma_min.to(t.device))
        return (t_normalized * (self.num_timesteps - 1)).long().clamp(0, self.num_timesteps - 1)

    def perturb_input(self, x_real, timesteps):
        t_normalized = timesteps.float() / (self.num_timesteps - 1)
        sigmas = self._map_to_noise_levels(t_normalized)
        noise = torch.randn_like(x_real)
        noisy_imgs = x_real + sigmas.view(-1, 1, 1, 1) * noise
        score_target = -noise / sigmas.view(-1, 1, 1, 1)
        return noisy_imgs, score_target, {'sigmas': sigmas}

    def compute_loss(self, pred, target, additional_info):
        sigmas = additional_info['sigmas']
        loss = F.mse_loss(pred, target, reduction='none')
        return (loss * sigmas.view(-1, 1, 1, 1)**2).mean()

    def reverse_process(self, perturbed_input, model_pred, timesteps, additional_info):
        sigmas = additional_info['sigmas']
        recon_imgs = perturbed_input + sigmas.view(-1, 1, 1, 1)**2 * model_pred
        return torch.clamp(recon_imgs, -1.0, 1.0)

    def get_solver_fn(self):
        def score_fn(x_t, sigma):
            timestep = self.get_timesteps_from_time(self._map_to_noise_levels.inverse(sigma))
            return self.unet(x_t, timestep).sample / sigma.view(-1, 1, 1, 1)
        return score_fn
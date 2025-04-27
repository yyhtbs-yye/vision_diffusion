import torch
import torch.nn.functional as F
from .ddpm import BaseDiffusionModel
from vision.solvers.ode_solvers import flow_matching_ode

class PixelFlowMatchingModel(BaseDiffusionModel):
    """Minimal flow matching model with shared logic in base class."""

    def __init__(self, model_config, train_config, validation_config):
        super().__init__(model_config, train_config, validation_config)
        self.t_start = train_config.get('t_start', 0.0)
        self.t_end = train_config.get('t_end', 1.0)
        self.solver = flow_matching_ode
        self.solver_kwargs = {'t_start': self.t_start, 't_end': self.t_end}

    def get_timesteps_from_time(self, t):
        return t  # UNet accepts continuous timesteps in [0,1]

    def perturb_input(self, x_real, timesteps):
        t = timesteps.float() / (self.num_timesteps - 1)
        x1 = x_real
        x0 = torch.randn_like(x_real)
        z_t = (1 - t.view(-1, 1, 1, 1)) * x1 + t.view(-1, 1, 1, 1) * x0
        v_target = x0 - x1
        return z_t, v_target, {'t': t}

    def compute_loss(self, pred, target, additional_info=None):
        return F.mse_loss(pred, target)

    def reverse_process(self, perturbed_input, model_pred, timesteps, additional_info):
        t = additional_info['t']
        dt = -t.view(-1, 1, 1, 1)
        return perturbed_input + model_pred * dt

    def get_solver_fn(self):
        def vector_field_fn(x, t):
            timestep = self.get_timesteps_from_time(t)
            return self.unet(x, timestep).sample
        return vector_field_fn
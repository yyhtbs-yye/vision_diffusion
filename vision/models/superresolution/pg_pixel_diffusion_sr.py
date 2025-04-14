import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from copy import deepcopy
import math
from diffusers.models.unets import UNet2DModel
from diffusers import DDIMScheduler
from vision.visualizers.basic_visualizer import visualize_comparisons
from vision.dist_samplers.conditioned_basic_sampler import inverse_add_noise, DefaultSampler

class PGPixelDiffusionSRModel(pl.LightningModule):
    
    def __init__(self, model_config, optimizer_config, validation_config):
        super().__init__()
        self.save_hyperparameters(ignore=['model_config'])
        
        # Initialize configuration
        self.unet_config = model_config.get('unet_config', {})
        self.in_channels = model_config.get('in_channels', 3)
        self.unet_config['out_channels'] = self.in_channels
        self.unet_config['in_channels'] = self.in_channels * 2  # Concatenate noisy image and LR condition
        
        # UNet strategy: shared or independent
        self.unet_strategy = model_config.get('unet_strategy', 'shared')
        if self.unet_strategy not in ['shared', 'independent']:
            raise ValueError(f"unet_strategy must be 'shared' or 'independent', got {self.unet_strategy}")
        
        # Adaptive stage configuration based on scale
        self.total_scale = model_config.get('scale', 8)
        self.base_lr_size = model_config.get('base_lr_size', 16)
        
        # Validate scale
        if not (self.total_scale > 0 and (self.total_scale & (self.total_scale - 1)) == 0):
            raise ValueError(f"Total scale {self.total_scale} must be a positive power of 2")
        
        # Compute stages starting from base_lr_size
        self.num_stages = int(math.log2(self.total_scale)) + 1  # Include base_lr_size
        self.stage_scale = 2
        self.stages = []
        current_res = self.base_lr_size  # Start at base_lr_size (e.g., 16)
        for i in range(self.num_stages):
            stage = {
                'resolution': current_res,
                'epochs': model_config.get('epochs_per_stage', 50),
                'fade_epochs': model_config.get('fade_epochs', 10) if i < self.num_stages - 1 else 0
            }
            self.stages.append(stage)
            current_res *= self.stage_scale
        
        # Initialize UNet(s)
        if self.unet_strategy == 'shared':
            self.unet = UNet2DModel.from_config(self.unet_config)
            self.unets = [self.unet] * self.num_stages
        else:
            self.unets = nn.ModuleList([UNet2DModel.from_config(self.unet_config) for _ in range(self.num_stages)])
        
        # Schedulers
        self.train_scheduler_config = model_config.get('train_scheduler_config', {})
        self.test_scheduler_config = model_config.get('test_scheduler_config', {})
        self.train_scheduler = DDIMScheduler.from_config(self.train_scheduler_config)
        self.test_scheduler = DDIMScheduler.from_config(self.test_scheduler_config)

        self.current_stage = 0
        self.current_epoch_in_stage = 0
        self.alpha = 0.0

        # Sampling parameters
        self.eta = model_config.get('eta', None)
        self.num_inference_steps = model_config.get('num_inference_steps', 50)
        
        # Validation settings
        self.num_vis_samples = validation_config.get('num_vis_samples', 4)

        self.automatic_optimization = False

        # Training parameters
        self.learning_rate = optimizer_config.get('learning_rate', 1e-4)
        self.betas = optimizer_config.get('betas', [0.9, 0.999])
        self.weight_decay = optimizer_config.get('weight_decay', 0.0)
        self.use_ema = optimizer_config.get('use_ema', True)
        self.ema_decay = optimizer_config.get('ema_decay', 0.999)
        self.ema_start = optimizer_config.get('ema_start', 1000)
        
        # EMA model(s)
        if self.use_ema:
            if self.unet_strategy == 'shared':
                self.unet_ema = deepcopy(self.unet)
                self.unets_ema = [self.unet_ema]
            else:
                self.unets_ema = nn.ModuleList([deepcopy(unet) for unet in self.unets])
            for unet_ema in self.unets_ema:
                for param in unet_ema.parameters():
                    param.requires_grad = False
        
        # Initialize sampler lazily in forward/validation
        self.test_sampler = None

    def _get_current_unet(self):
        return self.unets[self.current_stage if self.unet_strategy == 'independent' else 0]

    def _get_current_unet_ema(self):
        return self.unets_ema[self.current_stage if self.unet_strategy == 'independent' else 0]

    def _update_ema(self):
        if not self.use_ema:
            return
        current_unet = self._get_current_unet()
        current_unet_ema = self._get_current_unet_ema()
        for ema_param, param in zip(current_unet_ema.parameters(), current_unet.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        for ema_buffer, buffer in zip(current_unet_ema.buffers(), current_unet.buffers()):
            ema_buffer.data.copy_(buffer.data)

    def _get_current_resolution(self):
        return self.stages[self.current_stage]['resolution']

    def _is_fading(self):
        stage = self.stages[self.current_stage]
        return self.current_epoch_in_stage < stage['fade_epochs']

    def _update_alpha(self):
        stage = self.stages[self.current_stage]
        if self._is_fading():
            self.alpha = min(self.current_epoch_in_stage / stage['fade_epochs'], 1.0)
        else:
            self.alpha = 1.0

    def forward(self, lqs, num_inference_steps=50, return_dict=True):
        batch_size, c, lq_h, lq_w = lqs.shape
        current_res = self._get_current_resolution()
        
        if lq_h != self.base_lr_size or lq_w != self.base_lr_size:
            raise ValueError(f"Input size {lq_h}x{lq_w} must match base_lr_size {self.base_lr_size}x{self.base_lr_size}")
        
        h, w = current_res, current_res
        uqs = F.interpolate(lqs, size=(h, w), mode='bilinear', align_corners=False)
        noise = torch.randn(batch_size, c, h, w, device=self.device, dtype=self.unets[0].dtype)

        # Initialize sampler for current stage
        sampler_unet = self._get_current_unet_ema() if (self.use_ema and self.global_step >= self.ema_start) else self._get_current_unet()
        sampler = DefaultSampler(sampler_unet, self.test_scheduler)
        
        generated_images = sampler.sample(
            noise, uqs,
            num_inference_steps=num_inference_steps,
            eta=self.eta,
            generator=None
        )
        generated_images = generated_images * 0.5 + 0.5

        # Fading: blend with previous stage's output
        if self._is_fading() and self.current_stage > 0:
            prev_res = self.stages[self.current_stage - 1]['resolution']
            uqs_prev = F.interpolate(lqs, size=(prev_res, prev_res), mode='bilinear', align_corners=False)
            noise_prev = torch.randn(batch_size, c, prev_res, prev_res, device=self.device, dtype=self.unets[0].dtype)
            prev_unet = self.unets[self.current_stage - 1 if self.unet_strategy == 'independent' else 0]
            prev_unet_ema = self.unets_ema[self.current_stage - 1 if self.unet_strategy == 'independent' else 0]
            sampler_unet_prev = prev_unet_ema if (self.use_ema and self.global_step >= self.ema_start) else prev_unet
            sampler_prev = DefaultSampler(sampler_unet_prev, self.test_scheduler)
            generated_prev = sampler_prev.sample(
                noise_prev, uqs_prev,
                num_inference_steps=num_inference_steps,
                eta=self.eta,
                generator=None
            )
            generated_prev = generated_prev * 0.5 + 0.5
            generated_prev = F.interpolate(generated_prev, size=(h, w), mode='bilinear', align_corners=False)
            generated_images = (1 - self.alpha) * generated_prev + self.alpha * generated_images
        
        if return_dict:
            return {'samples': generated_images}
        else:
            return generated_images

    def training_step(self, batch, batch_idx):
        real_imgs = batch['gt']
        lq_imgs = batch['lq']
        
        current_res = self._get_current_resolution()
        batch_size, c, _, _ = real_imgs.shape

        real_imgs_resized = F.interpolate(real_imgs, size=(current_res, current_res), mode='bicubic', align_corners=False)
        uqs = F.interpolate(lq_imgs, size=(current_res, current_res), mode='bilinear', align_corners=False)

        noise = torch.randn_like(real_imgs_resized)
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        noisy_imgs = self.train_scheduler.add_noise(real_imgs_resized, noise, timesteps)
        cnoisy_imgs = torch.cat((noisy_imgs, uqs), dim=1)

        if self.train_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.train_scheduler.config.prediction_type == "v_prediction":
            target = self.train_scheduler.get_velocity(real_imgs_resized, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.train_scheduler.config.prediction_type}")
        
        noise_pred = self._get_current_unet()(cnoisy_imgs, timesteps).sample
        loss = F.mse_loss(noise_pred.float(), target.float())

        # Fading: include loss from previous stage
        if self._is_fading() and self.current_stage > 0:
            prev_res = self.stages[self.current_stage - 1]['resolution']
            real_imgs_prev = F.interpolate(real_imgs, size=(prev_res, prev_res), mode='bicubic', align_corners=False)
            uqs_prev = F.interpolate(lq_imgs, size=(prev_res, prev_res), mode='bilinear', align_corners=False)
            noise_prev = torch.randn_like(real_imgs_prev)
            noisy_imgs_prev = self.train_scheduler.add_noise(real_imgs_prev, noise_prev, timesteps)
            cnoisy_imgs_prev = torch.cat((noisy_imgs_prev, uqs_prev), dim=1)
            prev_unet = self.unets[self.current_stage - 1 if self.unet_strategy == 'independent' else 0]
            noise_pred_prev = prev_unet(cnoisy_imgs_prev, timesteps).sample
            # Use consistent target for previous stage
            target_prev = noise_prev if self.train_scheduler.config.prediction_type == "epsilon" else \
                         self.train_scheduler.get_velocity(real_imgs_prev, noise_prev, timesteps)
            loss_prev = F.mse_loss(noise_pred_prev.float(), target_prev.float())
            loss = self.alpha * loss + (1 - self.alpha) * loss_prev

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("stage", float(self.current_stage), prog_bar=True)
        self.log("alpha", self.alpha, prog_bar=True)
        
        if self.use_ema and self.global_step >= self.ema_start:
            self._update_ema()
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.test_scheduler.set_timesteps(self.test_scheduler.config.num_train_timesteps)
        
        real_imgs = batch['gt']
        lq_imgs = batch['lq']
        
        current_res = self._get_current_resolution()
        batch_size, c, _, _ = real_imgs.shape
        
        real_imgs_resized = F.interpolate(real_imgs, size=(current_res, current_res), mode='bicubic', align_corners=False)
        uqs = F.interpolate(lq_imgs, size=(current_res, current_res), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            noise = torch.randn_like(real_imgs_resized)
            timesteps = torch.randint(
                0, self.test_scheduler.config.num_train_timesteps,
                (batch_size,), device=self.device
            ).long()
            noisy_imgs = self.test_scheduler.add_noise(real_imgs_resized, noise, timesteps)
            cnoisy_imgs = torch.cat((noisy_imgs, uqs), dim=1)
            
            unet_to_use = self._get_current_unet_ema() if (self.use_ema and self.global_step >= self.ema_start) else self._get_current_unet()
            noise_pred = unet_to_use(cnoisy_imgs, timesteps).sample
            
            img_mse = F.mse_loss(noise_pred, noise)
            self.log("val/img_mse", img_mse)
            
            if batch_idx == 0 and self.logger is not None:
                img_denoised = inverse_add_noise(noisy_imgs, noise_pred, timesteps, self.test_scheduler)
                cmp_dict = {
                    'real': real_imgs_resized[:self.num_vis_samples],
                    'recon': img_denoised[:self.num_vis_samples]
                }
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=cmp_dict,
                    keys=list(cmp_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),
                    prefix=f'val_stage_{self.current_stage}'
                )
                
                sampler_unet = self._get_current_unet_ema() if (self.use_ema and self.global_step >= self.ema_start) else self._get_current_unet()
                sampler = DefaultSampler(sampler_unet, self.test_scheduler)
                generated_imgs = sampler.sample(
                    torch.randn_like(uqs), uqs,
                    num_inference_steps=self.num_inference_steps,
                    eta=self.eta
                )
                gen_dict = {'gen': generated_imgs}
                visualize_comparisons(
                    logger=self.logger.experiment,
                    images_dict=gen_dict,
                    keys=list(gen_dict.keys()),
                    global_step=self.global_step,
                    wnb=(0.5, 0.5),
                    prefix=f'val_stage_{self.current_stage}'
                )
        
        return {"val_loss": img_mse}

    def on_train_epoch_end(self):
        stage = self.stages[self.current_stage]
        self._update_alpha()  # Update alpha before incrementing epoch
        self.current_epoch_in_stage += 1
        
        if self.current_epoch_in_stage >= stage['epochs']:
            self.current_stage = min(self.current_stage + 1, len(self.stages) - 1)
            self.current_epoch_in_stage = 0
            self.alpha = 0.0
            # Reconfigure optimizer for new stage
            if self.unet_strategy == 'independent':
                self.trainer.optimizers = [self.configure_optimizers()]

    def configure_optimizers(self):
        if self.unet_strategy == 'shared':
            params = self.unet.parameters()
        else:
            params = self.unets[self.current_stage].parameters() if self.current_stage < len(self.unets) else []
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=tuple(self.betas),
            weight_decay=self.weight_decay
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        if self.unet_strategy == 'shared':
            checkpoint['unet'] = self.unet.state_dict()
            if self.use_ema:
                checkpoint['unet_ema'] = self.unet_ema.state_dict()
        else:
            checkpoint['unets'] = [unet.state_dict() for unet in self.unets]
            if self.use_ema:
                checkpoint['unets_ema'] = [unet_ema.state_dict() for unet_ema in self.unets_ema]
        checkpoint['current_stage'] = self.current_stage
        checkpoint['current_epoch_in_stage'] = self.current_epoch_in_stage
        checkpoint['alpha'] = self.alpha
        # Save scheduler states
        checkpoint['train_scheduler'] = self.train_scheduler.state_dict()
        checkpoint['test_scheduler'] = self.test_scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.unet_strategy == 'shared':
            if 'unet' in checkpoint:
                self.unet.load_state_dict(checkpoint['unet'])
            if self.use_ema and 'unet_ema' in checkpoint:
                self.unet_ema.load_state_dict(checkpoint['unet_ema'])
        else:
            if 'unets' in checkpoint:
                for i, state_dict in enumerate(checkpoint['unets']):
                    self.unets[i].load_state_dict(state_dict)
            if self.use_ema and 'unets_ema' in checkpoint:
                for i, state_dict in enumerate(checkpoint['unets_ema']):
                    self.unets_ema[i].load_state_dict(state_dict)
        self.current_stage = checkpoint.get('current_stage', 0)
        self.current_epoch_in_stage = checkpoint.get('current_epoch_in_stage', 0)
        self.alpha = checkpoint.get('alpha', 0.0)
        # Load scheduler states
        if 'train_scheduler' in checkpoint:
            self.train_scheduler.load_state_dict(checkpoint['train_scheduler'])
        if 'test_scheduler' in checkpoint:
            self.test_scheduler.load_state_dict(checkpoint['test_scheduler'])
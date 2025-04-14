import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention import Attention
from typing import Dict, Optional, Tuple

class VAEDiscriminatorConfig(ConfigMixin):
    """
    Configuration class for VAEDiscriminator, aligned with Diffusers' VAE config structure.
    """
    model_type = "vae_discriminator"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "DownDiscBlock2D",
            "DownDiscBlock2D",
            "DownDiscBlock2D",
            "DownDiscBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        attention_head_dim: Optional[int] = 8,
        add_attention: bool = True,
        sample_size: int = 256,
    ):
        self.in_channels = in_channels
        self.down_block_types = down_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.norm_num_groups = norm_num_groups
        self.act_fn = act_fn
        self.attention_head_dim = attention_head_dim
        self.add_attention = add_attention
        self.sample_size = sample_size

class DownDiscBlock2D(nn.Module):
    """
    Downsampling block for the discriminator, comparable to DownEncoderBlock2D in the VAE.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        add_attention: bool = False,
        num_attention_heads: int = 8,
        attention_head_dim: Optional[int] = 32,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        
        # First layer does the downsampling
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(norm_num_groups, out_channels),
                nn.SiLU() if act_fn == "silu" else nn.LeakyReLU(0.2)
            )
        )
        
        # Additional layers for feature extraction (similar to ResNet blocks but simpler)
        in_channels = out_channels
        for _ in range(num_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(norm_num_groups, out_channels),
                    nn.SiLU() if act_fn == "silu" else nn.LeakyReLU(0.2)
                )
            )
        
        # Add attention if specified - use Diffusers' Attention
        self.add_attention = add_attention
        if add_attention:
            self.attention = Attention(
                query_dim=out_channels,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                bias=True,
                out_bias=True,
            )
            self.norm = nn.GroupNorm(norm_num_groups, out_channels)
        else:
            self.attention = None
            self.norm = None
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        if self.add_attention:
            # Reshape for attention
            batch, channel, height, width = x.shape
            residual = x
            x = self.norm(x)
            x = x.reshape(batch, channel, height * width).transpose(1, 2)
            x = self.attention(x)
            x = x.transpose(1, 2).reshape(batch, channel, height, width)
            x = x + residual
            
        return x


class MidDiscBlock2D(nn.Module):
    """
    Middle block for the discriminator, comparable to the mid block in the VAE.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_attention: bool = True,
        num_attention_heads: int = 8,
        attention_head_dim: Optional[int] = 32,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(norm_num_groups, out_channels)
        self.act1 = nn.SiLU() if act_fn == "silu" else nn.LeakyReLU(0.2)
        
        self.add_attention = add_attention
        if add_attention:
            self.attention = Attention(
                query_dim=out_channels,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                bias=True,
                out_bias=True,
            )
            self.norm_attn = nn.GroupNorm(norm_num_groups, out_channels)
        else:
            self.attention = None
            self.norm_attn = None
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(norm_num_groups, out_channels)
        self.act2 = nn.SiLU() if act_fn == "silu" else nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        if self.add_attention:
            # Reshape for attention
            batch, channel, height, width = x.shape
            residual = x
            x = self.norm_attn(x)
            x = x.reshape(batch, channel, height * width).transpose(1, 2)
            x = self.attention(x)
            x = x.transpose(1, 2).reshape(batch, channel, height, width)
            x = x + residual
            
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        return x


class FeedForward(nn.Module):
    """
    Simple feed-forward network with residual connection.
    """
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        act_fn: str = "silu",
    ):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        inner_dim = int(dim * mult)
        
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.SiLU() if act_fn == "silu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class VAEDiscriminator(ModelMixin, ConfigMixin):
    """
    VAE Discriminator model compatible with Diffusers' architecture patterns.
    
    This discriminator follows a similar structure to the VAE encoder but with 
    modifications suitable for a discriminator network.
    """
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "DownDiscBlock2D",
            "DownDiscBlock2D",
            "DownDiscBlock2D",
            "DownDiscBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        attention_head_dim: Optional[int] = 8,
        add_attention: bool = True,
        sample_size: int = 256,
    ):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList([])
        
        output_channel = block_out_channels[0]
        
        # Build encoder
        down_block_types = down_block_types
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            down_block = DownDiscBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block,
                add_attention=add_attention and (i == len(block_out_channels) - 2 or is_final_block),
                attention_head_dim=attention_head_dim,
                norm_num_groups=norm_num_groups,
                act_fn=act_fn,
            )
            
            self.down_blocks.append(down_block)
        
        # Middle block
        self.mid_block = MidDiscBlock2D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            add_attention=add_attention,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )
        
        # Final prediction layer
        self.conv_out = nn.Conv2d(block_out_channels[-1], 1, kernel_size=4, stride=1, padding=1)
        
        # Calculate final resolution for adaptive averaging
        self.sample_size = sample_size
        self.num_down_blocks = len(down_block_types)
        # Each downsampling reduces resolution by factor of 2
        self.final_resolution = sample_size // (2 ** self.num_down_blocks)
        
    def forward(self, x: torch.FloatTensor, return_features: bool = False) -> Dict[str, torch.FloatTensor]:
        """
        Forward pass of the VAE Discriminator.
        
        Args:
            x (torch.FloatTensor): Input images [batch_size, in_channels, height, width]
            return_features (bool): Whether to return intermediate features
            
        Returns:
            Dict[str, torch.FloatTensor]: Model outputs including discriminator score
        """
        # Initial convolution
        h = self.conv_in(x)
        
        # Store intermediate features if needed
        if return_features:
            features = [h]
        
        # Down blocks
        for down_block in self.down_blocks:
            h = down_block(h)
            if return_features:
                features.append(h)
        
        # Middle block
        h = self.mid_block(h)
        if return_features:
            features.append(h)
        
        # Final prediction
        logits = self.conv_out(h)
        
        # Create output dict
        output = {"logits": logits}
        if return_features:
            output["features"] = features
            
        return output
    
    @classmethod
    def from_config(cls, config):
        """Creates a VAEDiscriminator from a configuration."""
        if isinstance(config, dict):
            return cls(**config)
        
        return cls(
            in_channels=config.in_channels,
            down_block_types=config.down_block_types,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            norm_num_groups=config.norm_num_groups,
            attention_head_dim=config.attention_head_dim,
            add_attention=config.add_attention,
            sample_size=config.sample_size,
        )

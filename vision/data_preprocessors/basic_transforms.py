import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional, Union, List

class SimpleTransform:
    """Collection of transforms for training and inference.
    
    This class provides easy access to standard transforms used in  training,
    including normalization, resizing, and data augmentation options.
    """
    
    @staticmethod
    def get_train_transform(
        image_size: Union[int, Tuple[int, int]] = 256,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        horizontal_flip: bool = True,
        flip_prob: float = 0.5
    ) -> transforms.Compose:
        """Get standard transforms for training.
        
        Args:
            image_size: Target image size (single int or tuple)
            normalize: Whether to normalize images
            mean: Channel means for normalization
            std: Channel stds for normalization
            horizontal_flip: Whether to apply random horizontal flips
            flip_prob: Probability of horizontal flip
            
        Returns:
            Composed transform pipeline
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
            
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        
        if horizontal_flip:
            transform_list.insert(1, transforms.RandomHorizontalFlip(flip_prob))
            
        if normalize:
            transform_list.append(transforms.Normalize(mean, std))
            
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_val_transform(
        image_size: Union[int, Tuple[int, int]] = 256,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ) -> transforms.Compose:
        """Get standard transforms for validation/inference.
        
        Args:
            image_size: Target image size (single int or tuple)
            normalize: Whether to normalize images
            mean: Channel means for normalization
            std: Channel stds for normalization
            
        Returns:
            Composed transform pipeline
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
            
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        
        if normalize:
            transform_list.append(transforms.Normalize(mean, std))
            
        return transforms.Compose(transform_list)
    
    @staticmethod
    def denormalize(
        tensor: torch.Tensor,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ) -> torch.Tensor:
        """Denormalize a tensor back to [0, 1] range.
        
        Args:
            tensor: Normalized input tensor
            mean: Channel means used for normalization
            std: Channel stds used for normalization
            
        Returns:
            Denormalized tensor with values in [0, 1]
        """
        device = tensor.device
        mean = torch.tensor(mean, device=device).view(-1, 1, 1)
        std = torch.tensor(std, device=device).view(-1, 1, 1)
        
        # Handle different tensor dimensions
        if tensor.dim() == 4:  # Batch of images
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        
        # Denormalize
        tensor = tensor * std + mean
        
        # Clamp values to [0, 1]
        return torch.clamp(tensor, 0, 1)
    
    @staticmethod
    def tensor_to_image(
        tensor: torch.Tensor,
        denormalize: bool = True,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    ) -> torch.Tensor:
        """Convert tensor to image format (denormalize and scale to [0, 255]).
        
        Args:
            tensor: Input tensor
            denormalize: Whether to denormalize the tensor
            mean: Channel means for denormalization
            std: Channel stds for denormalization
            
        Returns:
            Tensor with values in [0, 255] range
        """
        if denormalize:
            tensor = SimpleTransform.denormalize(tensor, mean, std)
        return tensor * 255
import torch

class Crop:
    def __init__(self, keys, crop_size, is_pad_zeros=True, random_crop=False):
        self.keys = keys
        self.crop_size = crop_size if isinstance(crop_size, list) else [crop_size, crop_size]
        self.is_pad_zeros = is_pad_zeros
        self.random_crop = random_crop
        
    def __call__(self, results):
        for key in self.keys:
            # Assuming input is already a tensor in CHW format
            img = results[key]
            c, h, w = img.shape
            crop_h, crop_w = self.crop_size
            
            if self.random_crop:
                # Random crop
                x_offset = torch.randint(0, max(0, w - crop_w) + 1, (1,)).item()
                y_offset = torch.randint(0, max(0, h - crop_h) + 1, (1,)).item()
            else:
                # Center crop
                x_offset = max(0, (w - crop_w) // 2)
                y_offset = max(0, (h - crop_h) // 2)
            
            # Calculate crop boundaries
            crop_y1, crop_y2 = y_offset, min(y_offset + crop_h, h)
            crop_x1, crop_x2 = x_offset, min(x_offset + crop_w, w)
            
            # Crop the image (tensor is in CHW format)
            crop_img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Pad with zeros if needed
            if self.is_pad_zeros and (crop_img.shape[1] < crop_h or crop_img.shape[2] < crop_w):
                pad_h = max(0, crop_h - crop_img.shape[1])
                pad_w = max(0, crop_w - crop_img.shape[2])
                # Pad tensor (CHW format)
                crop_img = torch.nn.functional.pad(
                    crop_img,
                    (0, pad_w, 0, pad_h),  # padding left, right, top, bottom
                    mode='constant',
                    value=0
                )
            
            results[key] = crop_img
            
        return results

class Normalize:
    """
    Normalize image tensor with mean and std.
    
    Args:
        keys (list[str]): The keys of images to be normalized.
        mean (list[float]): Mean values for each channel.
        std (list[float]): Std values for each channel.
    """
    
    def __init__(self, keys, mean, std):
        self.keys = keys
        self.mean = mean
        self.std = std
        
    def __call__(self, results):
        for key in self.keys:
            if key not in results:
                continue
                
            img = results[key]
            
            # Simple normalization for tensor in CHW format
            mean = torch.tensor(self.mean, dtype=torch.float32, device=img.device).view(-1, 1, 1)
            std = torch.tensor(self.std, dtype=torch.float32, device=img.device).view(-1, 1, 1)
            img = (img - mean) / std
            
            results[key] = img
                
        return results
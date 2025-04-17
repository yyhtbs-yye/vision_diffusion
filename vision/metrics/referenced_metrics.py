import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    """
    return ssim(img1, img2, multichannel=True, data_range=img1.max() - img1.min())

def calculate_mse(img1, img2):
    """
    Calculate Mean Squared Error (MSE) between two images.
    """
    return np.mean((img1 - img2) ** 2)

# Example usage:
# img1 and img2 should be numpy arrays of the same shape.
# psnr_value = calculate_psnr(img1, img2)
# ssim_value = calculate_ssim(img1, img2)
# mse_value = calculate_mse(img1, img2)
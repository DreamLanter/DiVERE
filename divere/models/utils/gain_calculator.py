"""
RGB Gain Calculator for Color Constancy Models
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import LinearRegression


def kernelP(I: np.ndarray) -> np.ndarray:
    """
    Simplified kernel function: kernel(r, g, b) -> (r, g, b, 1)
    Only includes RGB gain terms for simple linear mapping
    """
    return np.transpose((
        I[:, 0], I[:, 1], I[:, 2], 
        np.repeat(1, np.shape(I)[0])
    ))


def get_mapping_func(image1: np.ndarray, image2: np.ndarray) -> LinearRegression:
    """
    Computes the simplified linear mapping between two images
    Only RGB gain terms, no cross-terms or higher-order terms
    """
    image1_flat = np.reshape(image1, [-1, 3])
    image2_flat = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1_flat), image2_flat)
    return m


def apply_mapping_func(image: np.ndarray, m: LinearRegression) -> np.ndarray:
    """
    Applies the simplified linear mapping to an image
    """
    sz = image.shape
    image_flat = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image_flat))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result


def calculate_rgb_gains_from_images(input_image: np.ndarray, 
                                   output_image: np.ndarray,
                                   method: str = 'simple_ratio') -> Tuple[float, float, float]:
    """
    Calculate RGB gains from input and output images
    
    Args:
        input_image: Input image array (H, W, 3) in range [0, 1]
        output_image: Output image array (H, W, 3) in range [0, 1]
        method: Method to calculate gains ('simple_ratio', 'log_ratio', 'linear_mapping')
    
    Returns:
        Tuple of (r_gain, g_gain, b_gain) in log space
    """
    if method == 'linear_mapping':
        return _calculate_gains_linear_mapping(input_image, output_image)
    elif method == 'simple_ratio':
        return _calculate_gains_simple_ratio(input_image, output_image)
    elif method == 'log_ratio':
        return _calculate_gains_log_ratio(input_image, output_image)
    else:
        raise ValueError(f"Unknown method: {method}")


def _calculate_gains_linear_mapping(input_image: np.ndarray, 
                                   output_image: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate RGB gains using simplified linear mapping method
    Only RGB gain terms, no cross-terms
    """
    # Get mapping function
    mapping_func = get_mapping_func(input_image, output_image)
    
    # Apply mapping to neutral gray to get gain factors
    neutral_gray = np.ones((100, 100, 3)) * 0.5  # 50% gray
    mapped_gray = apply_mapping_func(neutral_gray, mapping_func)
    
    # Calculate gains as the ratio of mapped to original
    gains = np.mean(mapped_gray, axis=(0, 1)) / 0.5
    
    # Convert to log space
    log_gains = np.log10(gains)
    
    # Clip gains to reasonable range
    log_gains = np.clip(log_gains, -1.0, 1.0)
    
    return tuple(log_gains)


def _calculate_gains_simple_ratio(input_image: np.ndarray, 
                                 output_image: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate RGB gains using simple ratio method
    """
    # Calculate mean values for each channel
    input_means = np.mean(input_image, axis=(0, 1))
    output_means = np.mean(output_image, axis=(0, 1))
    
    # Avoid division by zero
    input_means = np.maximum(input_means, 1e-8)
    
    # Calculate gains as ratio
    gains = output_means / input_means
    
    # Convert to log space
    log_gains = np.log10(gains)
    
    # Clip gains to reasonable range
    log_gains = np.clip(log_gains, -1.0, 1.0)
    
    return tuple(log_gains)


def _calculate_gains_log_ratio(input_image: np.ndarray, 
                              output_image: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate RGB gains using log ratio method
    """
    # Calculate mean values for each channel
    input_means = np.mean(input_image, axis=(0, 1))
    output_means = np.mean(output_image, axis=(0, 1))
    
    # Avoid log of zero
    input_means = np.maximum(input_means, 1e-8)
    output_means = np.maximum(output_means, 1e-8)
    
    # Calculate gains as log ratio
    log_gains = np.log10(output_means) - np.log10(input_means)
    
    # Clip gains to reasonable range
    log_gains = np.clip(log_gains, -1.0, 1.0)
    
    return tuple(log_gains)


def estimate_illuminant_from_gains(gains: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Estimate illuminant color from RGB gains
    
    Args:
        gains: RGB gains in log space
        
    Returns:
        Estimated illuminant RGB values
    """
    # Convert from log space to linear space
    gain_factors = np.power(10, np.array(gains))
    
    # The illuminant is the inverse of the gains (what we're correcting for)
    illuminant = 1.0 / gain_factors
    
    # Normalize to have G=1 as reference
    illuminant = illuminant / illuminant[1]
    
    return tuple(illuminant) 
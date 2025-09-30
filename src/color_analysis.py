"""
Color analysis module for explicit color extraction and comparison.
Extracts dominant colors from masked regions and compares distributions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from skimage import color as skcolor


class ColorAnalyzer(nn.Module):
    """
    Analyzes colors in masked regions of images.
    Extracts dominant colors, filters shadows/highlights, and compares distributions.
    """
    
    def __init__(
        self,
        n_colors: int = 5,
        shadow_percentile: float = 10.0,
        highlight_percentile: float = 90.0,
        min_pixels: int = 100
    ):
        """
        Args:
            n_colors: Number of dominant colors to extract
            shadow_percentile: Percentile below which to consider shadows
            highlight_percentile: Percentile above which to consider highlights
            min_pixels: Minimum number of pixels required for analysis
        """
        super().__init__()
        self.n_colors = n_colors
        self.shadow_percentile = shadow_percentile
        self.highlight_percentile = highlight_percentile
        self.min_pixels = min_pixels
    
    def extract_masked_pixels(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor
    ) -> np.ndarray:
        """
        Extract pixels from masked region.
        
        Args:
            image: RGB image (B, 3, H, W) in [0, 1]
            mask: Binary mask (B, 1, H, W) in [0, 1]
            
        Returns:
            Masked pixels as numpy array (N, 3) in [0, 1]
        """
        # Convert to numpy
        image_np = image.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        
        # Get first batch item
        image_np = image_np[0].transpose(1, 2, 0)  # (H, W, 3)
        mask_np = mask_np[0, 0]  # (H, W)
        
        # Extract masked pixels
        mask_bool = mask_np > 0.5
        masked_pixels = image_np[mask_bool]  # (N, 3)
        
        return masked_pixels
    
    def filter_shadows_highlights(
        self, 
        pixels: np.ndarray,
        is_dark_garment: bool = False,
        is_light_garment: bool = False
    ) -> np.ndarray:
        """
        Filter out shadows and highlights unless they're the base color.
        
        Args:
            pixels: RGB pixels (N, 3) in [0, 1]
            is_dark_garment: True if garment is dark (keep shadows)
            is_light_garment: True if garment is light (keep highlights)
            
        Returns:
            Filtered pixels (M, 3) where M <= N
        """
        if len(pixels) == 0:
            return pixels
        
        # Convert to LAB for better luminance separation
        lab_pixels = skcolor.rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
        
        # L channel ranges from 0 (black) to 100 (white)
        l_values = lab_pixels[:, 0]
        
        # Calculate percentiles
        l_shadow = np.percentile(l_values, self.shadow_percentile)
        l_highlight = np.percentile(l_values, self.highlight_percentile)
        
        # Create filter mask
        keep_mask = np.ones(len(pixels), dtype=bool)
        
        # Filter shadows (unless dark garment)
        if not is_dark_garment:
            keep_mask &= l_values > l_shadow
        
        # Filter highlights (unless light garment)
        if not is_light_garment:
            keep_mask &= l_values < l_highlight
        
        return pixels[keep_mask]
    
    def detect_garment_type(self, pixels: np.ndarray) -> Tuple[bool, bool]:
        """
        Detect if garment is dark or light based on luminance.
        
        Args:
            pixels: RGB pixels (N, 3) in [0, 1]
            
        Returns:
            (is_dark, is_light) tuple
        """
        if len(pixels) == 0:
            return False, False
        
        # Convert to LAB
        lab_pixels = skcolor.rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
        mean_l = np.mean(lab_pixels[:, 0])
        
        # Dark if mean L < 30, light if mean L > 70
        is_dark = mean_l < 30.0
        is_light = mean_l > 70.0
        
        return is_dark, is_light
    
    def extract_dominant_colors(
        self, 
        pixels: np.ndarray,
        n_colors: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract dominant colors using K-means clustering.
        
        Args:
            pixels: RGB pixels (N, 3) in [0, 1]
            n_colors: Number of colors to extract (default: self.n_colors)
            
        Returns:
            (colors, weights) where colors is (K, 3) and weights is (K,)
        """
        if n_colors is None:
            n_colors = self.n_colors
        
        if len(pixels) < self.min_pixels:
            # Not enough pixels, return mean color
            mean_color = np.mean(pixels, axis=0, keepdims=True)
            return mean_color, np.array([1.0])
        
        # Limit number of colors to number of pixels
        n_colors = min(n_colors, len(pixels))
        
        # K-means clustering in LAB space (more perceptual)
        lab_pixels = skcolor.rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(lab_pixels)
        
        # Get cluster centers (in LAB)
        centers_lab = kmeans.cluster_centers_  # (K, 3)
        
        # Convert back to RGB
        centers_rgb = skcolor.lab2rgb(centers_lab.reshape(-1, 1, 3)).reshape(-1, 3)
        centers_rgb = np.clip(centers_rgb, 0, 1)
        
        # Calculate weights (proportion of pixels in each cluster)
        weights = np.bincount(labels, minlength=n_colors) / len(labels)
        
        # Sort by weight (descending)
        sort_idx = np.argsort(-weights)
        colors = centers_rgb[sort_idx]
        weights = weights[sort_idx]
        
        return colors, weights
    
    def compute_color_distribution(
        self,
        pixels: np.ndarray,
        n_bins: int = 32
    ) -> np.ndarray:
        """
        Compute color histogram distribution.
        
        Args:
            pixels: RGB pixels (N, 3) in [0, 1]
            n_bins: Number of bins per channel
            
        Returns:
            Histogram (n_bins * 3,)
        """
        if len(pixels) == 0:
            return np.zeros(n_bins * 3)
        
        # Compute histograms for each channel
        hists = []
        for c in range(3):
            hist, _ = np.histogram(pixels[:, c], bins=n_bins, range=(0, 1))
            hist = hist / (np.sum(hist) + 1e-8)  # Normalize
            hists.append(hist)
        
        return np.concatenate(hists)
    
    def analyze_image(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict:
        """
        Analyze colors in a masked image.
        
        Args:
            image: RGB image (B, 3, H, W) in [0, 1]
            mask: Binary mask (B, 1, H, W) in [0, 1]
            
        Returns:
            Dictionary with color analysis results
        """
        # Extract masked pixels
        pixels = self.extract_masked_pixels(image, mask)
        
        if len(pixels) < self.min_pixels:
            # Not enough pixels for analysis
            return {
                'dominant_colors': np.array([[0.5, 0.5, 0.5]]),
                'color_weights': np.array([1.0]),
                'n_colors': 1,
                'mean_color': np.array([0.5, 0.5, 0.5]),
                'histogram': np.zeros(32 * 3),
                'is_dark': False,
                'is_light': False
            }
        
        # Detect garment type
        is_dark, is_light = self.detect_garment_type(pixels)
        
        # Filter shadows/highlights
        filtered_pixels = self.filter_shadows_highlights(
            pixels, is_dark, is_light
        )
        
        # If filtering removed too many pixels, use original
        if len(filtered_pixels) < self.min_pixels:
            filtered_pixels = pixels
        
        # Extract dominant colors
        colors, weights = self.extract_dominant_colors(filtered_pixels)
        
        # Compute mean color
        mean_color = np.average(colors, axis=0, weights=weights)
        
        # Compute histogram
        histogram = self.compute_color_distribution(filtered_pixels)
        
        return {
            'dominant_colors': colors,
            'color_weights': weights,
            'n_colors': len(colors),
            'mean_color': mean_color,
            'histogram': histogram,
            'is_dark': is_dark,
            'is_light': is_light,
            'n_pixels': len(pixels),
            'n_filtered_pixels': len(filtered_pixels)
        }
    
    def compare_colors(
        self,
        analysis_still: Dict,
        analysis_onmodel: Dict
    ) -> Dict:
        """
        Compare color distributions between still and on-model.
        
        Args:
            analysis_still: Color analysis of still image
            analysis_onmodel: Color analysis of on-model image
            
        Returns:
            Dictionary with comparison metrics
        """
        # Mean color difference (Euclidean in RGB)
        mean_diff = np.linalg.norm(
            analysis_still['mean_color'] - analysis_onmodel['mean_color']
        )
        
        # Histogram correlation
        hist_corr = np.corrcoef(
            analysis_still['histogram'],
            analysis_onmodel['histogram']
        )[0, 1]
        
        if np.isnan(hist_corr):
            hist_corr = 0.0
        
        # Dominant color comparison (Hungarian matching)
        colors_still = analysis_still['dominant_colors']
        colors_onmodel = analysis_onmodel['dominant_colors']
        
        # For simplicity, compare top colors
        n_compare = min(len(colors_still), len(colors_onmodel), 3)
        dominant_diff = 0.0
        for i in range(n_compare):
            diff = np.linalg.norm(colors_still[i] - colors_onmodel[i])
            weight = (analysis_still['color_weights'][i] + 
                     analysis_onmodel['color_weights'][i]) / 2
            dominant_diff += diff * weight
        
        return {
            'mean_color_diff': float(mean_diff),
            'histogram_corr': float(hist_corr),
            'dominant_color_diff': float(dominant_diff),
            'n_colors_still': analysis_still['n_colors'],
            'n_colors_onmodel': analysis_onmodel['n_colors']
        }
    
    def forward(
        self,
        still: torch.Tensor,
        onmodel: torch.Tensor,
        mask_still: torch.Tensor,
        mask_onmodel: torch.Tensor
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Analyze and compare colors in both images.
        
        Args:
            still: Still image (B, 3, H, W)
            onmodel: On-model image (B, 3, H, W)
            mask_still: Still mask (B, 1, H, W)
            mask_onmodel: On-model mask (B, 1, H, W)
            
        Returns:
            (analysis_still, analysis_onmodel, comparison)
        """
        analysis_still = self.analyze_image(still, mask_still)
        analysis_onmodel = self.analyze_image(onmodel, mask_onmodel)
        comparison = self.compare_colors(analysis_still, analysis_onmodel)
        
        return analysis_still, analysis_onmodel, comparison

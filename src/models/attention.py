"""
Cross-attention block with 2D sinusoidal positional encoding.
"""
import math
import threading
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, temperature: float = 10000.0) -> torch.Tensor:
    """
    Generate 2D sinusoidal position embeddings.
    
    Args:
        embed_dim: Embedding dimension (must be even)
        grid_size: Grid size (assumes square grid)
        temperature: Temperature for sinusoidal encoding
        
    Returns:
        Position embeddings of shape (grid_size*grid_size, embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    
    # Create grid coordinates
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)  # Shape: (2, grid_size, grid_size)
    
    # Normalize coordinates to [0, 1]
    grid = grid / (grid_size - 1)
    
    # Create frequency bands
    omega = torch.arange(embed_dim // 4, dtype=torch.float32)
    omega = 1.0 / (temperature ** (omega / (embed_dim // 4)))
    
    # Apply sinusoidal encoding
    pos_embed = []
    for i in range(2):  # For h and w coordinates
        coords = grid[i].flatten()  # Shape: (grid_size*grid_size,)
        coords = coords.unsqueeze(-1) * omega.unsqueeze(0)  # Broadcasting
        pos_embed.append(torch.sin(coords))
        pos_embed.append(torch.cos(coords))
    
    pos_embed = torch.cat(pos_embed, dim=1)  # Shape: (grid_size*grid_size, embed_dim)
    return pos_embed


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention with 2D positional encoding."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pos_embed: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pos_embed = use_pos_embed
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Thread-safe position embedding cache
        self._pos_embed_cache = {}
        self._cache_lock = threading.RLock()
    
    def _get_pos_embed(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Get or create position embeddings for given spatial dimensions with thread safety."""
        if not self.use_pos_embed:
            return None
        
        key = (height, width, str(device))
        
        with self._cache_lock:
            if key not in self._pos_embed_cache:
                # Clear cache if it gets too large to prevent memory leaks
                if len(self._pos_embed_cache) > 100:
                    self._pos_embed_cache.clear()
                
                # Assume square grid for simplicity
                grid_size = max(height, width)
                pos_embed = get_2d_sincos_pos_embed(self.embed_dim, grid_size)
                
                # Crop to actual size if needed
                if height != width:
                    total_positions = height * width
                    pos_embed = pos_embed[:total_positions]
                
                self._pos_embed_cache[key] = pos_embed.to(device)
        
        return self._pos_embed_cache[key]
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.
        
        Args:
            query: Query tensor (B, C, H_q, W_q)
            key: Key tensor (B, C, H_k, W_k)
            value: Value tensor (B, C, H_v, W_v)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as query
        """
        B, C, H_q, W_q = query.shape
        _, _, H_k, W_k = key.shape
        _, _, H_v, W_v = value.shape
        
        assert H_k == H_v and W_k == W_v, "Key and value must have same spatial dimensions"
        
        # Reshape to sequence format: (B, C, H, W) -> (B, H*W, C)
        q = query.view(B, C, H_q * W_q).transpose(1, 2)  # (B, H_q*W_q, C)
        k = key.view(B, C, H_k * W_k).transpose(1, 2)    # (B, H_k*W_k, C)
        v = value.view(B, C, H_v * W_v).transpose(1, 2)  # (B, H_v*W_v, C)
        
        # Add positional embeddings
        if self.use_pos_embed:
            pos_q = self._get_pos_embed(H_q, W_q, query.device)
            pos_k = self._get_pos_embed(H_k, W_k, key.device)
            
            if pos_q is not None:
                q = q + pos_q.unsqueeze(0)
            if pos_k is not None:
                k = k + pos_k.unsqueeze(0)
        
        # Project to Q, K, V
        Q = self.q_proj(q)  # (B, H_q*W_q, C)
        K = self.k_proj(k)  # (B, H_k*W_k, C)
        V = self.v_proj(v)  # (B, H_v*W_v, C)
        
        # Reshape for multi-head attention
        Q = Q.view(B, H_q * W_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, H_q*W_q, head_dim)
        K = K.view(B, H_k * W_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, H_k*W_k, head_dim)
        V = V.view(B, H_v * W_v, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, H_v*W_v, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, H_q*W_q, H_k*W_k)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, H_q*W_q, head_dim)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H_q * W_q, self.embed_dim)
        output = self.out_proj(attn_output)
        
        # Reshape back to spatial format
        output = output.transpose(1, 2).view(B, C, H_q, W_q)
        
        return output


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with residual connection and layer norm."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pos_embed: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.attention = MultiHeadCrossAttention(
            embed_dim, num_heads, dropout, use_pos_embed
        )
        
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with residual connections.
        
        Args:
            query: Query tensor (B, C, H, W)
            key: Key tensor (B, C, H, W)
            value: Value tensor (B, C, H, W)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor with same shape as query
        """
        B, C, H, W = query.shape
        
        # Self-attention with residual connection
        query_flat = query.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        attn_output = self.attention(query, key, value, attention_mask)
        attn_output_flat = attn_output.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # First residual connection
        query_flat = self.norm1(query_flat + attn_output_flat)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(query_flat)
        output_flat = self.norm2(query_flat + ffn_output)
        
        # Reshape back to spatial format
        output = output_flat.transpose(1, 2).view(B, C, H, W)
        
        return output

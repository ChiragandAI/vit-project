import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Single Transformer Encoder block used in Vision Transformer.

    Structure:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Input shape:
        (B, tokens, emb_dim)

    Output shape:
        (B, tokens, emb_dim)
    """

    def __init__(self, emb_dim=128, num_heads=4, mlp_dim=256, dropout=0.0):
        super().__init__()

        # Layer normalization before attention
        self.norm1 = nn.LayerNorm(emb_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True  # keeps tensor format (B, tokens, emb_dim)
        )

        # Layer normalization before feed-forward network
        self.norm2 = nn.LayerNorm(emb_dim)

        # Feed-forward network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass through the transformer block.
        """

        # Self-attention with residual connection
        attn_input = self.norm1(x)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        x = x + attn_output

        # Feed-forward network with residual connection
        mlp_input = self.norm2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output

        return x
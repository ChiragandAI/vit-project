import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts an input image into a sequence of patch embeddings.

    Example:
        Input image  : (B, 3, 32, 32)
        Patch size   : 4
        Output tokens: (B, 64, emb_dim)

    Explanation:
        32x32 image with patch_size=4 → 8x8 patches → 64 tokens
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, emb_dim=128):
        super().__init__()

        # Store important hyperparameters
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        # Number of patches in the image
        # Example: (32 / 4)^2 = 64
        self.num_patches = (img_size // patch_size) ** 2

        # Convolution used to extract non-overlapping patches
        # kernel_size = patch_size  → size of each patch
        # stride = patch_size       → ensures patches do not overlap
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Forward pass.

        Input:
            x → (B, C, H, W)

        Output:
            tokens → (B, num_patches, emb_dim)
        """

        # Apply patch projection
        # Output shape: (B, emb_dim, H_patch, W_patch)
        x = self.projection(x)

        # Flatten spatial dimensions
        # (B, emb_dim, H_patch * W_patch)
        x = x.flatten(2)

        # Rearrange to transformer format
        # (B, num_patches, emb_dim)
        x = x.transpose(1, 2)

        return x
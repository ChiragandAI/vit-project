import torch
import torch.nn as nn

from models.patch_embeddings import PatchEmbedding
from models.transformer_block import TransformerBlock


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT)

    Pipeline:
        image
        ↓
        patch embedding
        ↓
        add CLS token
        ↓
        add positional embeddings
        ↓
        transformer encoder layers
        ↓
        CLS token output
        ↓
        classification head
    """

    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        emb_dim=128,
        depth=6,
        num_heads=4,
        mlp_dim=256,
        dropout=0.0
    ):
        super().__init__()

        # Patch embedding module
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            emb_dim=emb_dim
        )

        # Number of patch tokens
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token
        # Shape: (1, 1, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Positional embeddings for patches + CLS token
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_dim)
        )

        # Dropout after embeddings
        self.dropout = nn.Dropout(dropout)

        # Stack of transformer encoder blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(
                emb_dim=emb_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Final normalization before classification
        self.norm = nn.LayerNorm(emb_dim)

        # Classification head
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):

        # Step 1: convert image → patch tokens
        x = self.patch_embed(x)
        # shape: (B, num_patches, emb_dim)

        batch_size = x.shape[0]

        # Step 2: expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Step 3: prepend CLS token
        x = torch.cat((cls_tokens, x), dim=1)
        # shape: (B, num_patches + 1, emb_dim)

        # Step 4: add positional embeddings
        x = x + self.pos_embed

        # Step 5: dropout
        x = self.dropout(x)

        # Step 6: transformer encoder
        for block in self.transformer:
            x = block(x)

        # Step 7: final normalization
        x = self.norm(x)

        # Step 8: extract CLS token
        cls_output = x[:, 0]

        # Step 9: classification
        logits = self.head(cls_output)

        return logits
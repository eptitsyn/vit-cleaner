import torch
import torch.nn as nn
from transformers import ViTModel
from vit_pytorch.vit import Transformer


class DocumentCleaningViT(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224-in21k", img_size=224, patch_size=16):
        super().__init__()

        # Load the pretrained ViT model as encoder
        self.encoder = ViTModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size

        # Calculate number of patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Positional embeddings for decoder
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_size) * 0.02)

        # Transformer Decoder from vit_pytorch
        self.decoder = Transformer(
            dim=hidden_size,
            depth=6,  # Number of transformer layers
            heads=12,  # Number of attention heads
            dim_head=hidden_size // 12,  # Dimension per head
            mlp_dim=hidden_size * 4,  # Feedforward dimension
            dropout=0.1
        )

        # Learnable query embeddings for the decoder
        self.query_embeddings = nn.Parameter(torch.randn(1, self.num_patches, hidden_size) * 0.02)

        # Final projection to patch pixels with layer norm
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(hidden_size),  # Added layer norm
            nn.Linear(hidden_size, patch_size * patch_size * 3),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Get encoder features
        encoder_output = self.encoder(x).last_hidden_state

        # Remove CLS token and use as memory
        memory = encoder_output[:, 1:, :]  # [B, num_patches, hidden_size]

        # Add positional information to memory
        memory = memory + self.pos_embedding

        # Prepare query with positional information
        query = self.query_embeddings.expand(batch_size, -1, -1)
        query = query + self.pos_embedding

        # Concatenate for self-attention in decoder
        decoder_input = torch.cat([query, memory], dim=1)  # [B, 2*num_patches, hidden_size]

        # Pass through decoder
        decoded_features = self.decoder(decoder_input)

        # Extract query part
        decoded_features = decoded_features[:, :self.num_patches, :]

        # Project to pixels
        patches = self.to_pixels(decoded_features)

        # Reshape into image
        h = w = self.img_size // self.patch_size
        reconstructed = patches.view(
            batch_size, h, w, self.patch_size, self.patch_size, 3
        )

        # Rearrange patches to form the final image
        reconstructed = reconstructed.permute(0, 5, 1, 3, 2, 4).contiguous()
        reconstructed = reconstructed.view(batch_size, 3, self.img_size, self.img_size)

        # Use tanh activation for output
        reconstructed = torch.tanh(reconstructed)

        return reconstructed

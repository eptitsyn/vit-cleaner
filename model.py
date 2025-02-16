import torch
import torch.nn as nn
from transformers import ViTModel


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
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_size))

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=12,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=6
        )

        # Learnable query embeddings for the decoder
        self.query_embeddings = nn.Parameter(torch.randn(1, self.num_patches, hidden_size))

        # Final projection to patch pixels
        self.to_pixels = nn.Sequential(
            nn.Linear(hidden_size, patch_size * patch_size * 3),
            nn.GELU()
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Get encoder features [batch_size, num_patches + 1, hidden_size]
        encoder_output = self.encoder(x).last_hidden_state

        # Remove CLS token and use as memory
        memory = encoder_output[:, 1:, :]  # [batch_size, num_patches, hidden_size]

        # Add positional information to memory
        memory = memory + self.pos_embedding

        # Prepare query with positional information
        query = self.query_embeddings.expand(batch_size, -1, -1)
        query = query + self.pos_embedding

        # Decode
        decoded_features = self.decoder(
            query,
            memory
        )

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

        # Use tanh instead of sigmoid for better range
        reconstructed = torch.tanh(reconstructed)

        return reconstructed

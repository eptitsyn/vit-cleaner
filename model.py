import torch
import torch.nn as nn
from transformers import ViTModel


class DocumentCleaningViT(nn.Module):
    def __init__(
        self,
        pretrained_model_name="google/vit-base-patch16-224-in21k",
        img_size=224,
        patch_size=16,
        decoder_dim=768,
        decoder_depth=6,
        decoder_heads=8,
        decoder_dim_head=64
    ):
        super().__init__()

        # Load the pretrained ViT model as encoder
        self.encoder = ViTModel.from_pretrained(pretrained_model_name)

        # Get encoder dimensions
        encoder_dim = self.encoder.config.hidden_size

        # Calculate number of patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Extract patch embedding layer from encoder
        self.to_patch_embedding = self.encoder.embeddings.patch_embeddings

        # Decoder components
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_depth
        )

        # Learnable query embeddings for the decoder
        self.query_embeddings = nn.Parameter(torch.randn(1, self.num_patches, decoder_dim))
        self.decoder_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, decoder_dim))

        # Final projection to patch pixels
        pixel_values_per_patch = patch_size * patch_size * 3
        self.to_pixels = nn.Sequential(
            nn.Linear(decoder_dim, pixel_values_per_patch),
            nn.GELU()
        )

    def forward(self, img):
        batch_size = img.shape[0]

        # Get patch embeddings and add position embeddings
        patches = self.to_patch_embedding(img)

        # Pass through encoder
        encoder_output = self.encoder(
            img,
            output_hidden_states=True,
            return_dict=True
        )

        # Get encoder features (excluding CLS token)
        encoder_features = encoder_output.last_hidden_state[:, 1:, :]

        # Project encoder features to decoder dimension
        memory = self.enc_to_dec(encoder_features)

        # Prepare decoder query embeddings
        query = self.query_embeddings.expand(batch_size, -1, -1)
        query = query + self.decoder_pos_emb

        # Decode
        decoded_features = self.decoder(
            query,
            memory
        )

        # Project to patch pixels
        patches = self.to_pixels(decoded_features)

        # Reshape into image
        h = w = self.img_size // self.patch_size
        reconstructed = patches.view(
            batch_size, h, w, self.patch_size, self.patch_size, 3
        )

        # Rearrange patches to form the final image
        reconstructed = reconstructed.permute(0, 5, 1, 3, 2, 4).contiguous()
        reconstructed = reconstructed.view(batch_size, 3, self.img_size, self.img_size)

        # Ensure output is in [0,1] range
        reconstructed = torch.sigmoid(reconstructed)

        return reconstructed

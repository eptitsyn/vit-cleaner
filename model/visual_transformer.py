import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from einops import rearrange


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, tgt, memory):
        # Self attention
        tgt2 = self.norm1(tgt)
        q = k = v = tgt2
        tgt2, _ = self.self_attn(q, k, v)
        tgt = tgt + self.dropout1(tgt2)

        # Cross attention
        tgt2 = self.norm2(tgt)
        tgt2, _ = self.cross_attn(tgt2, memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        # Feed forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(decoder_layer.linear1.in_features)

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return self.norm(tgt)


class DocumentCleaningViT(nn.Module):
    def __init__(
        self,
        pretrained_model_name="google/vit-base-patch16-224-in21k",
        img_size=224,
        patch_size=16,
        in_channels=3,
        dropout=0.1,
        attention_dropout=0.1,
        freeze_backbone=False,
        decoder_layers=6,
        decoder_heads=8
    ):
        super().__init__()

        # Initialize the ViT configuration
        self.config = ViTConfig(
            image_size=img_size,
            patch_size=patch_size,
            num_channels=in_channels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=attention_dropout,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_size=768,
            num_hidden_layers=12,
            output_hidden_states=True
        )

        # Load pre-trained ViT model
        self.vit = ViTModel.from_pretrained(
            pretrained_model_name,
            config=self.config,
            add_pooling_layer=False
        )

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Initialize transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=768,  # Same as ViT hidden size
            nhead=decoder_heads,
            dim_feedforward=3072,  # Same as ViT intermediate size
            dropout=dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, decoder_layers)

        # Final projection to image space
        self.final_layer = nn.Sequential(
            nn.Linear(768, patch_size * patch_size * in_channels),
            nn.Tanh()  # Ensure output is in [-1, 1] range
        )

        # Learnable query embeddings for the decoder
        num_patches = (img_size // patch_size) ** 2
        self.query_embeddings = nn.Parameter(torch.randn(1, num_patches, 768))

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels

    def forward(self, x):
        # Forward pass through ViT
        outputs = self.vit(x)
        encoder_hidden_states = outputs.last_hidden_state

        # Get batch size for expanding query embeddings
        batch_size = x.shape[0]
        query_embeddings = self.query_embeddings.expand(batch_size, -1, -1)

        # Decode with transformer decoder
        decoded_features = self.decoder(query_embeddings, encoder_hidden_states)

        # Project to image space
        decoded_patches = self.final_layer(decoded_features)

        # Reshape patches to image
        decoded_patches = rearrange(
            decoded_patches,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.img_size//self.patch_size,
            w=self.img_size//self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.in_channels
        )

        return decoded_patches

    def load_pretrained(self, checkpoint_path):
        """Load pretrained weights for the entire model"""
        self.load_state_dict(torch.load(checkpoint_path))

    def save_pretrained(self, save_path):
        """Save the entire model's weights"""
        torch.save(self.state_dict(), save_path)


class DocumentCleaningLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * mse

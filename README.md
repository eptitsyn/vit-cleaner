# Document Artifact Cleaning with Vision Transformer

This project implements a Vision Transformer (ViT) model for cleaning artifacts from scanned document images. The model leverages the pre-trained ViT from Hugging Face Transformers library and is specifically fine-tuned for removing various types of artifacts such as noise, stains, folds, and other imperfections while preserving the original text and document content.

## Features

- Pre-trained Vision Transformer backbone from Hugging Face
- Custom decoder for image reconstruction
- Flexible architecture with configurable parameters
- Combined L1 and MSE loss for better image quality
- Supports various image sizes (default: 224x224)

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
import torch
from model.visual_transformer import DocumentCleaningViT, DocumentCleaningLoss

# Initialize the model
model = DocumentCleaningViT(
    pretrained_model_name="google/vit-base-patch16-224-in21k",  # Pre-trained model to use
    img_size=224,          # Input image size
    patch_size=16,         # Size of the patches
    in_channels=3,         # Number of input channels
    dropout=0.1,           # Dropout rate
    attention_dropout=0.1,  # Attention dropout rate
    freeze_backbone=False  # Whether to freeze the pre-trained backbone
)

# Initialize the loss function
criterion = DocumentCleaningLoss(alpha=0.8)  # alpha controls the balance between L1 and MSE loss

# Load an image (should be a tensor of shape [batch_size, channels, height, width])
# image = ...

# Process the image
with torch.no_grad():
    cleaned_image = model(image)
```

## Model Architecture

The model consists of two main components:

1. **Vision Transformer Encoder**: Pre-trained ViT model from Hugging Face that processes the input image into a sequence of patch embeddings
2. **Custom Decoder**: Converts the processed embeddings back into a cleaned image through a series of linear layers

## Training

To train the model on your own dataset:

1. Prepare pairs of corrupted and clean document images
2. Use the provided `DocumentCleaningLoss` which combines L1 and MSE losses
3. Example training loop:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        corrupted_images, clean_images = batch

        # Forward pass
        cleaned_images = model(corrupted_images)
        loss = criterion(cleaned_images, clean_images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Saving and Loading Models

The model provides convenient methods for saving and loading weights:

```python
# Save model
model.save_pretrained('path/to/save/model.pth')

# Load model
model.load_pretrained('path/to/saved/model.pth')
```

## License

MIT License

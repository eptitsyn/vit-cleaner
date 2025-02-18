import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor
from torchvision.transforms.functional import to_tensor, to_pil_image
from tqdm import tqdm

from model import DocumentCleaningViT


class DocumentCleaner:
    def __init__(
        self,
        model_path,
        model_name="google/vit-base-patch16-224-in21k",
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DocumentCleaningViT(model_name)

        # Load model weights
        state_dict = torch.load(model_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = {k.replace('model.', ''): v for k, v in state_dict['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        # Get model parameters
        self.patch_size = self.model.patch_size
        self.window_size = self.model.img_size
        self.overlap = self.window_size // 4  # 25% overlap

    def preprocess_image(self, image):
        """Preprocess image using the model's image processor"""
        return self.image_processor(image, return_tensors="pt")["pixel_values"][0]

    def process_window(self, window):
        """Process a single window through the model"""
        with torch.no_grad():
            window = window.unsqueeze(0).to(self.device)
            output = self.model(window)
            return output.squeeze(0).cpu()

    def blend_edges(self, img1, img2, overlap, direction='horizontal'):
        """Blend overlapping edges using linear weights"""
        if direction == 'horizontal':
            weight1 = torch.linspace(1, 0, overlap).view(1, 1, overlap, 1)
            weight2 = torch.linspace(0, 1, overlap).view(1, 1, overlap, 1)
        else:  # vertical
            weight1 = torch.linspace(1, 0, overlap).view(1, 1, 1, overlap)
            weight2 = torch.linspace(0, 1, overlap).view(1, 1, 1, overlap)

        return img1 * weight1 + img2 * weight2

    def clean_image(self, image_path, output_path=None):
        """Clean document image of any size using sliding window approach"""
        # Load and convert image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Convert to tensor
        img_tensor = to_tensor(image)

        # Calculate number of windows needed
        stride = self.window_size - self.overlap
        n_h = max(1, (height - self.overlap) // stride)
        n_w = max(1, (width - self.overlap) // stride)

        # Pad image if needed
        pad_h = max(0, (n_h * stride + self.overlap) - height)
        pad_w = max(0, (n_w * stride + self.overlap) - width)
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        # Create output tensor
        output = torch.zeros_like(img_tensor)
        counts = torch.zeros_like(img_tensor)

        # Process each window
        with torch.no_grad():
            for i in tqdm(range(n_h), desc="Processing rows"):
                for j in range(n_w):
                    # Extract window
                    y = i * stride
                    x = j * stride
                    window = img_tensor[:, y:y+self.window_size, x:x+self.window_size]

                    # Process window
                    processed = self.process_window(self.preprocess_image(to_pil_image(window)))

                    # Add processed window to output
                    output[:, y:y+self.window_size, x:x+self.window_size] += processed
                    counts[:, y:y+self.window_size, x:x+self.window_size] += 1

        # Average overlapping regions
        output = output / counts

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :height, :width]

        # Convert to PIL Image
        output_image = to_pil_image(output)

        # Save if output path provided
        if output_path:
            output_image.save(output_path)

        return output_image


def main():
    parser = argparse.ArgumentParser(description="Clean document artifacts from images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output image or directory")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Initialize cleaner
    cleaner = DocumentCleaner(
        model_path=args.model_path,
        device=args.device
    )

    # Process input path
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if input_path.is_file():
        # Single file processing
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        cleaner.clean_image(input_path, output_path)
    else:
        # Directory processing
        output_path.mkdir(parents=True, exist_ok=True)
        image_files = list(input_path.glob("*.[jp][pn][g]"))

        for img_path in tqdm(image_files, desc="Processing images"):
            out_file = output_path / img_path.name
            cleaner.clean_image(img_path, out_file)


if __name__ == "__main__":
    main()

import random
from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
import cv2
from tqdm import tqdm


class CoffeeStain(DualTransform):
    def __init__(self, num_stains=(1, 3), alpha=(0.1, 0.3), always_apply=False, p=0.4):
        super().__init__(always_apply, p)
        self.num_stains = num_stains
        self.alpha = alpha

    def apply(self, img, **params):
        num_stains = random.randint(*self.num_stains)
        mask = np.zeros_like(img)

        for _ in range(num_stains):
            center = (random.randint(0, img.shape[1]), random.randint(0, img.shape[0]))
            axes = (random.randint(20, 50), random.randint(20, 50))
            angle = random.uniform(0, 360)
            color = (random.randint(19, 40), random.randint(69, 90), random.randint(139, 160))

            cv2.ellipse(mask, center, axes, angle, 0, 360, color, -1)

        alpha = random.uniform(*self.alpha)
        return cv2.addWeighted(img, 1, mask, alpha, 0)

    def get_transform_init_args_names(self):
        return ("num_stains", "alpha")


class PaperWrinkle(DualTransform):
    def __init__(self, num_wrinkles=(3, 8), intensity=(2, 5), always_apply=False, p=0.7):
        super().__init__(always_apply, p)
        self.num_wrinkles = num_wrinkles
        self.intensity = intensity

    def apply(self, img, **params):
        num_wrinkles = random.randint(*self.num_wrinkles)
        displacement = np.zeros(img.shape[:2], dtype=np.float32)

        for _ in range(num_wrinkles):
            x1, y1 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
            x2, y2 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
            cv2.line(displacement, (x1, y1), (x2, y2), 1, thickness=random.randint(2, 5))

        displacement = cv2.GaussianBlur(displacement, (21, 21), 11)

        rows, cols = np.indices(img.shape[:2])
        intensity = random.uniform(*self.intensity)

        displacement_x = (displacement * intensity).astype(np.float32)
        displacement_y = (displacement * intensity).astype(np.float32)

        return cv2.remap(
            img,
            (cols + displacement_x).astype(np.float32),
            (rows + displacement_y).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    def get_transform_init_args_names(self):
        return ("num_wrinkles", "intensity")


class PaperFold(DualTransform):
    def __init__(self, thickness=(2, 4), shadow_intensity=0.3, always_apply=False, p=0.6):
        super().__init__(always_apply, p)
        self.thickness = thickness
        self.shadow_intensity = shadow_intensity

    def apply(self, img, **params):
        shadow = np.zeros_like(img)

        # Horizontal fold
        if random.random() < 0.7:
            y = random.randint(img.shape[0]//4, 3*img.shape[0]//4)
            thickness = random.randint(*self.thickness)
            cv2.line(shadow, (0, y), (img.shape[1], y), (50, 50, 50), thickness)

        # Vertical fold
        if random.random() < 0.7:
            x = random.randint(img.shape[1]//4, 3*img.shape[1]//4)
            thickness = random.randint(*self.thickness)
            cv2.line(shadow, (x, 0), (x, img.shape[0]), (50, 50, 50), thickness)

        return cv2.addWeighted(img, 1, shadow, self.shadow_intensity, 0)

    def get_transform_init_args_names(self):
        return ("thickness", "shadow_intensity")


class DocumentArtifactGenerator:
    def __init__(
        self,
        output_dir,
        img_size=224,
        num_samples=1000,
        font_dir=None,
        background_dir=None
    ):
        self.output_dir = Path(output_dir)
        self.clean_dir = self.output_dir / 'clean'
        self.corrupted_dir = self.output_dir / 'corrupted'
        self.img_size = img_size
        self.num_samples = num_samples

        # Create output directories
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.corrupted_dir.mkdir(parents=True, exist_ok=True)

        # Load fonts
        self.font_dir = font_dir or '/usr/share/fonts/truetype'
        self.fonts = self._load_fonts()

        # Initialize augmentations
        self.clean_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.8),
                A.ColorJitter(p=0.8),
            ], p=0.5),
        ])

        self.artifact_transform = A.Compose([
            CoffeeStain(p=0.4),
            PaperWrinkle(p=0.7),
            PaperFold(p=0.6),
            A.GaussNoise(var_limit=(5, 50), p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.6),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2),
                    contrast_limit=(-0.2, 0.2),
                    p=0.8
                ),
                A.ColorJitter(p=0.8),
            ], p=0.7),
            A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
        ])

    def _load_fonts(self):
        """Load available fonts from the font directory"""
        fonts = []
        font_sizes = range(12, 36, 4)

        for font_path in Path(self.font_dir).rglob('*.ttf'):
            try:
                for size in font_sizes:
                    fonts.append(ImageFont.truetype(str(font_path), size))
            except (OSError, IOError) as e:
                print(f"Could not load font {font_path}: {e}")
                continue

        if not fonts:
            raise RuntimeError(f"No usable fonts found in {self.font_dir}")

        return fonts

    def _generate_clean_document(self):
        """Generate a clean document image with random text"""
        # Create blank image
        img = Image.new('RGB', (self.img_size, self.img_size), 'white')
        draw = ImageDraw.Draw(img)

        # Add random text
        font = random.choice(self.fonts)
        margin = 20
        y = margin

        # Generate random paragraphs
        while y < self.img_size - margin:
            # Generate random words for the text
            words = []
            text_length = random.randint(30, 60)
            for _ in range(text_length):
                word_len = random.randint(3, 8)
                word = ''.join(random.choices(
                    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                    k=word_len
                ))
                words.append(word)

            text = ' '.join(words)
            draw.text((margin, y), text, fill='black', font=font)
            y += font.size + random.randint(5, 15)

        # Convert to numpy array for albumentations
        img_array = np.array(img)

        # Apply clean transforms
        transformed = self.clean_transform(image=img_array)
        return transformed['image']

    def _corrupt_image(self, img):
        """Apply random corruptions using albumentations"""
        transformed = self.artifact_transform(image=img)
        return transformed['image']

    def generate_dataset(self):
        """Generate the complete dataset"""
        print(f"Generating {self.num_samples} document pairs...")

        for i in tqdm(range(self.num_samples)):
            # Generate clean document
            clean_img = self._generate_clean_document()

            # Generate corrupted version
            corrupted_img = self._corrupt_image(clean_img)

            # Save images
            filename = f"document_{i:05d}.png"
            cv2.imwrite(str(self.clean_dir / filename), cv2.cvtColor(clean_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(self.corrupted_dir / filename), cv2.cvtColor(corrupted_img, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic document dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of image pairs to generate")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (square)")
    parser.add_argument("--font_dir", type=str, help="Directory containing .ttf fonts")
    parser.add_argument("--background_dir", type=str, help="Directory containing background textures")

    args = parser.parse_args()

    generator = DocumentArtifactGenerator(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        img_size=args.img_size,
        font_dir=args.font_dir,
        background_dir=args.background_dir
    )

    generator.generate_dataset()


if __name__ == "__main__":
    main()

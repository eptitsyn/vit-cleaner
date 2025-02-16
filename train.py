import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torchvision

from model.visual_transformer import DocumentCleaningViT, DocumentCleaningLoss


class DocumentCleaningDataset(Dataset):
    def __init__(self, clean_dir, corrupted_dir, transform=None):
        """
        Dataset for document cleaning.

        Args:
            clean_dir (str): Directory containing clean document images
            corrupted_dir (str): Directory containing corrupted document images
            transform: Albumentations transforms
        """
        self.clean_dir = clean_dir
        self.corrupted_dir = corrupted_dir
        self.transform = transform

        self.image_files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Load clean and corrupted images
        clean_path = os.path.join(self.clean_dir, img_name)
        corrupted_path = os.path.join(self.corrupted_dir, img_name)

        clean_img = np.array(Image.open(clean_path).convert('RGB'))
        corrupted_img = np.array(Image.open(corrupted_path).convert('RGB'))

        if self.transform:
            # Apply same transform to both images
            transformed = self.transform(image=clean_img, image1=corrupted_img)
            clean_img = transformed['image']
            corrupted_img = transformed['image1']

        return {
            'clean': clean_img,
            'corrupted': corrupted_img,
            'filename': img_name
        }


class DocumentCleaningModule(pl.LightningModule):
    def __init__(
        self,
        model_name="google/vit-base-patch16-224-in21k",
        img_size=224,
        patch_size=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        loss_alpha=0.8,
        train_batch_size=32,
        eval_batch_size=64,
        num_workers=4,
        train_clean_dir=None,
        train_corrupted_dir=None,
        val_clean_dir=None,
        val_corrupted_dir=None,
        log_dir='lightning_logs',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model with only relevant parameters
        model_kwargs = {
            'pretrained_model_name': model_name,
            'img_size': img_size,
            'patch_size': patch_size
        }
        self.model = DocumentCleaningViT(**model_kwargs)

        # Initialize loss
        self.criterion = DocumentCleaningLoss(alpha=loss_alpha)

        # Data transforms
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'image1': 'image'})

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean = batch['clean']
        corrupted = batch['corrupted']

        # Forward pass
        cleaned = self(corrupted)
        loss = self.criterion(cleaned, clean)

        # Log training metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            # Log sample images
            self._log_images(clean, corrupted, cleaned, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        clean = batch['clean']
        corrupted = batch['corrupted']

        # Forward pass
        cleaned = self(corrupted)
        loss = self.criterion(cleaned, clean)

        # Log validation metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            # Log sample images
            self._log_images(clean, corrupted, cleaned, 'val')

        return loss

    def _log_images(self, clean, corrupted, cleaned, prefix, num_images=4):
        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(clean.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(clean.device)

        clean = clean[:num_images] * std + mean
        corrupted = corrupted[:num_images] * std + mean
        cleaned = cleaned[:num_images] * std + mean

        # Clip values to [0, 1]
        clean = torch.clamp(clean, 0, 1)
        corrupted = torch.clamp(corrupted, 0, 1)
        cleaned = torch.clamp(cleaned, 0, 1)

        # Create image grid
        grid = torchvision.utils.make_grid(
            torch.cat([clean, corrupted, cleaned]),
            nrow=num_images
        )

        # Log to tensorboard
        self.logger.experiment.add_image(f'{prefix}_samples', grid, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def train_dataloader(self):
        train_dataset = DocumentCleaningDataset(
            clean_dir=self.hparams.train_clean_dir,
            corrupted_dir=self.hparams.train_corrupted_dir,
            transform=self.transform
        )
        return DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        val_dataset = DocumentCleaningDataset(
            clean_dir=self.hparams.val_clean_dir,
            corrupted_dir=self.hparams.val_corrupted_dir,
            transform=self.transform
        )
        return DataLoader(
            val_dataset,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DocumentCleaningModule")
        parser.add_argument('--model_name', type=str, default="google/vit-base-patch16-224-in21k")
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--patch_size', type=int, default=16)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--loss_alpha', type=float, default=0.8)
        parser.add_argument('--train_batch_size', type=int, default=32)
        parser.add_argument('--eval_batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--decoder_layers', type=int, default=6)
        parser.add_argument('--decoder_heads', type=int, default=8)
        return parent_parser


def main():
    parser = ArgumentParser()

    # Add program level args
    parser.add_argument('--train_clean_dir', type=str, required=True)
    parser.add_argument('--train_corrupted_dir', type=str, required=True)
    parser.add_argument('--val_clean_dir', type=str, required=True)
    parser.add_argument('--val_corrupted_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='lightning_logs')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default='auto')

    # Add model specific args
    parser = DocumentCleaningModule.add_model_specific_args(parser)

    args = parser.parse_args()

    # Initialize tensorboard logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name='document-cleaning'
    )

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='document-cleaning-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=args.accelerator,
        devices=args.devices
    )

    # Initialize model
    model = DocumentCleaningModule(**vars(args))

    # Train model
    trainer.fit(model)


if __name__ == '__main__':
    main()

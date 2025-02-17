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
import torch.nn as nn
from transformers import AutoImageProcessor

from model import DocumentCleaningViT
from utils import get_cosine_schedule_with_warmup


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


class DocumentCleaningDataset(Dataset):
    def __init__(self, clean_dir, corrupted_dir, image_processor=None):
        """
        Dataset for document cleaning.

        Args:
            clean_dir (str): Directory containing clean document images
            corrupted_dir (str): Directory containing corrupted document images
            image_processor: Image processor from transformers
        """
        self.clean_dir = clean_dir
        self.corrupted_dir = corrupted_dir
        self.image_processor = image_processor

        # Cache file lists
        self.image_files = sorted([
            f for f in os.listdir(clean_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Pre-compute file paths
        self.clean_paths = [os.path.join(clean_dir, f) for f in self.image_files]
        self.corrupted_paths = [os.path.join(corrupted_dir, f) for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load images
        clean_img = Image.open(self.clean_paths[idx]).convert('RGB')
        corrupted_img = Image.open(self.corrupted_paths[idx]).convert('RGB')

        # Process images using the model's preprocessor
        clean_processed = self.image_processor(clean_img, return_tensors="pt")["pixel_values"][0]
        corrupted_processed = self.image_processor(corrupted_img, return_tensors="pt")["pixel_values"][0]

        return {
            'clean': clean_processed,
            'corrupted': corrupted_processed,
            'filename': self.image_files[idx]
        }


class DocumentCleaningModule(pl.LightningModule):
    def __init__(
        self,
        model_name="google/vit-base-patch16-224-in21k",
        img_size=224,
        patch_size=16,
        learning_rate=5e-5,
        weight_decay=0.05,
        loss_alpha=0.7,
        train_batch_size=32,
        eval_batch_size=64,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
        warmup_steps=500,
        train_clean_dir=None,
        train_corrupted_dir=None,
        val_clean_dir=None,
        val_corrupted_dir=None,
        log_dir='lightning_logs',
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Get the image processor from the pretrained model
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        model_kwargs = {
            'pretrained_model_name': model_name,
            'img_size': img_size,
            'patch_size': patch_size
        }
        self.model = DocumentCleaningViT(**model_kwargs)

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Save hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_alpha = loss_alpha
        self.warmup_steps = warmup_steps

        # Initialize step counter for logging
        self.train_step_count = 0
        self.val_step_count = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean = batch['clean']
        corrupted = batch['corrupted']

        # Process images using the model's preprocessor
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Forward pass with preprocessed images
            cleaned = self(corrupted)
            # clean = clean * 2 - 1  # Scale to [-1, 1] to match tanh output

            l1_loss = self.l1_loss(cleaned, clean)
            mse_loss = self.mse_loss(cleaned, clean)
            loss = self.loss_alpha * l1_loss + (1 - self.loss_alpha) * mse_loss

        # Reduce logging frequency
        if self.global_step % 500 == 0:
            self.log('train/l1_loss', l1_loss, on_step=True, prog_bar=True)
            self.log('train/mse_loss', mse_loss, on_step=True, prog_bar=True)
            self.log('train/total_loss', loss, on_step=True, prog_bar=True)
            self._log_images(clean, corrupted, cleaned, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        clean = batch['clean']
        corrupted = batch['corrupted']

        # Forward pass
        cleaned = self(corrupted)

        # Scale targets to [-1, 1] to match tanh output
        clean = clean * 2 - 1

        # Compute losses
        l1_loss = self.l1_loss(cleaned, clean)
        mse_loss = self.mse_loss(cleaned, clean)
        loss = self.loss_alpha * l1_loss + (1 - self.loss_alpha) * mse_loss

        # Log validation metrics
        self.log('val/l1_loss', l1_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log sample images every 50 validation steps
        if self.val_step_count % 50 == 0:
            self._log_images(clean, corrupted, cleaned, 'val')

        self.val_step_count += 1
        return loss

    def _log_images(self, clean, corrupted, cleaned, prefix, num_images=4):
        # Scale back to [0, 1] for visualization
        cleaned = (cleaned + 1) / 2
        clean = (clean + 1) / 2

        # Take first num_images
        clean = clean[:num_images]
        corrupted = corrupted[:num_images]
        cleaned = cleaned[:num_images]

        # Create grid
        grid = torchvision.utils.make_grid(
            torch.cat([clean, corrupted, cleaned]),
            nrow=num_images
        )

        # Log to tensorboard
        self.logger.experiment.add_image(
            f'{prefix}/samples',
            grid,
            global_step=self.train_step_count if prefix == 'train' else self.val_step_count
        )

    def configure_optimizers(self):
        # Separate parameters for different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            },
        }

    def train_dataloader(self):
        train_dataset = DocumentCleaningDataset(
            clean_dir=self.hparams.train_clean_dir,
            corrupted_dir=self.hparams.train_corrupted_dir,
            image_processor=self.image_processor
        )
        return DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            drop_last=True
        )

    def val_dataloader(self):
        val_dataset = DocumentCleaningDataset(
            clean_dir=self.hparams.val_clean_dir,
            corrupted_dir=self.hparams.val_corrupted_dir,
            image_processor=self.image_processor
        )
        return DataLoader(
            val_dataset,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            prefetch_factor=self.hparams.prefetch_factor
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DocumentCleaningModule")
        parser.add_argument('--model_name', type=str, default="google/vit-base-patch16-224-in21k")
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--patch_size', type=int, default=16)
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        parser.add_argument('--weight_decay', type=float, default=0.05)
        parser.add_argument('--loss_alpha', type=float, default=0.7)
        parser.add_argument('--train_batch_size', type=int, default=32)
        parser.add_argument('--eval_batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=12)
        parser.add_argument('--pin_memory', type=bool, default=True)
        parser.add_argument('--persistent_workers', type=bool, default=True)
        parser.add_argument('--prefetch_factor', type=int, default=3)
        parser.add_argument('--warmup_steps', type=int, default=500)
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
        monitor='val/total_loss',
        dirpath='checkpoints',
        filename='document-cleaning-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize trainer with performance optimizations
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision='16-mixed',
        accumulate_grad_batches=4,
        # strategy='ddp_find_unused_parameters_false',
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Initialize model
    model = DocumentCleaningModule(**vars(args))

    # Train model
    trainer.fit(model)


if __name__ == '__main__':
    main()

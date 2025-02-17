import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision
import torch.nn as nn
from transformers import AutoImageProcessor

from model import DocumentCleaningViT


class DocumentCleaningDataset(Dataset):
    def __init__(self, clean_dir, corrupted_dir, image_processor=None):
        self.clean_dir = clean_dir
        self.corrupted_dir = corrupted_dir
        self.image_processor = image_processor

        self.image_files = sorted([
            f for f in os.listdir(clean_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.clean_paths = [os.path.join(clean_dir, f) for f in self.image_files]
        self.corrupted_paths = [os.path.join(corrupted_dir, f) for f in self.image_files]

    def __getitem__(self, idx):
        clean_img = Image.open(self.clean_paths[idx]).convert('RGB')
        corrupted_img = Image.open(self.corrupted_paths[idx]).convert('RGB')

        clean_processed = self.image_processor(clean_img, return_tensors="pt")["pixel_values"][0]
        corrupted_processed = self.image_processor(corrupted_img, return_tensors="pt")["pixel_values"][0]

        return {
            'clean': clean_processed,
            'corrupted': corrupted_processed
        }

    def __len__(self):
        return len(self.image_files)


class DocumentCleaningModule(pl.LightningModule):
    def __init__(
        self,
        model_name="google/vit-base-patch16-224-in21k",
        img_size=224,
        patch_size=16,
        learning_rate=5e-5,
        train_batch_size=32,
        eval_batch_size=64,
        num_workers=12,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DocumentCleaningViT(model_name, img_size, patch_size)
        self.mse_loss = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clean = batch['clean']
        corrupted = batch['corrupted']

        with torch.cuda.amp.autocast():
            cleaned = self.model(corrupted)
            clean = clean * 2 - 1  # Scale to [-1, 1] to match tanh output
            loss = self.mse_loss(cleaned, clean)

        self.log('train_loss', loss, on_step=True, prog_bar=True)

        if self.global_step % 500 == 0:
            self._log_images(clean, corrupted, cleaned, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        clean = batch['clean']
        corrupted = batch['corrupted']

        cleaned = self.model(corrupted)
        clean = clean * 2 - 1
        loss = self.mse_loss(cleaned, clean)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.global_step % 500 == 0:
            self._log_images(clean, corrupted, cleaned, 'val')

        return loss

    def _log_images(self, clean, corrupted, cleaned, prefix, num_images=4):
        cleaned = (cleaned + 1) / 2
        clean = (clean + 1) / 2

        grid = torchvision.utils.make_grid(
            torch.cat([clean[:num_images], corrupted[:num_images], cleaned[:num_images]]),
            nrow=num_images
        )

        self.logger.experiment.add_image(f'{prefix}/samples', grid, self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

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
            pin_memory=True,
            persistent_workers=True,
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
            pin_memory=True,
            persistent_workers=True
        )


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_clean_dir', type=str, required=True)
    parser.add_argument('--train_corrupted_dir', type=str, required=True)
    parser.add_argument('--val_clean_dir', type=str, required=True)
    parser.add_argument('--val_corrupted_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)

    args = parser.parse_args()

    logger = TensorBoardLogger('lightning_logs', name='document-cleaning')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='document-cleaning-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=2,
        precision='16-mixed',
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    model = DocumentCleaningModule(**vars(args))
    trainer.fit(model)


if __name__ == '__main__':
    main()

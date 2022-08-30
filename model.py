from typing import Any, Union, List

import logging

from transformers import AutoModel
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import torch

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class RestorationModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, num_classes: int) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.save_hyperparameters(cfg)

        self.encoder = AutoModel.from_pretrained(cfg.encoder.name)
        self.head = self._construct_head()

    def _construct_head(self) -> nn.Module:
        if self.hparams.model.head.architecture == 'mlp':
            mlp_config = self.hparams.model.head.mlp

            head = nn.Sequential()
            prev_layer_dim = self.encoder.config.hidden_size
            for i in range(mlp_config.num_layers - 1):
                layer = nn.Linear(prev_layer_dim, mlp_config.num_hiddens)
                prev_layer_dim = layer.out_features

                if mlp_config.nonlinearity == 'relu':
                    nonlinearity = nn.ReLU()
                else:
                    raise ValueError('unknown nonlinearity')

                head.add_module(f'cls{i}', layer)
                head.add_module(f'cls{i}_nonlinearity', nonlinearity)
            
            head.add_module('cls_pred', nn.Linear(prev_layer_dim, self.num_classes))

            return head
        
        else:
            raise ValueError('unknown head architecture')

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        y_hat = self.head(z)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        y_hat = self.head(z)
        
        val_loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', val_loss)

        pred = torch.argmax(F.softmax(y_hat, dim=1), dim=1)

        return pred, y

    # FIXME shouldn't it be validation_step_end instead? https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#validating-with-dataparallel
    def validation_epoch_end(self, validation_step_outputs) -> None:
        print(validation_step_outputs)
        all_preds = torch.stack(validation_step_outputs)

        # TODO calculate and log metrics


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        y_hat = self.head(z)
        test_loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.training.learning_rate)
        return optimizer

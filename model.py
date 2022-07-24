from typing import Any

import logging

from transformers import AutoModel
import pytorch_lightning as pl
from torch import nn
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
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.training.learning_rate)
        return super().configure_optimizers()

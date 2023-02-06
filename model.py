from typing import Any, List, Dict

import logging

from transformers import AutoModel, BatchEncoding
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from torch import nn
import torch.nn.functional as F
import torch

import numpy as np
import numpy.typing as npt
from sklearn.metrics import precision_recall_fscore_support

from omegaconf import DictConfig

from datatypes import ValidationEpochOutputs

logger = logging.getLogger(__name__)


class RestorationModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, num_classes: int) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.save_hyperparameters(cfg)

        self.loss_weights = torch.tensor(
            [self.hparams.trainer.zero_class_weight] \
            + [1.0 for _ in range(self.num_classes - 1)]
        )
        self.loss = nn.CrossEntropyLoss(self.loss_weights)

        self.encoder = AutoModel.from_pretrained(cfg.model.encoder.name)
        self.head = self._construct_head()

        self.head.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

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

    def _calculate_metrics(self, preds: npt.NDArray[np.int], golds: npt.NDArray[np.int]) -> Dict[str, Any]:
        labels = list(range(1, self.trainer.datamodule.num_classes))
        pr_micro, rc_micro, f1_micro, _ = \
            precision_recall_fscore_support(golds, preds, average='micro', zero_division=0, labels=labels)
        pr_macro, rc_macro, f1_macro, _ = \
            precision_recall_fscore_support(golds, preds, average='macro', zero_division=0, labels=labels)
        pr_weighted, rc_weighted, f1_weighted, _ = \
            precision_recall_fscore_support(golds, preds, average='weighted', zero_division=0, labels=labels)
        pr_per_label, rc_per_label, f1_per_label, _ = \
            precision_recall_fscore_support(golds, preds, average=None, zero_division=0, labels=labels)

        metrics = {
                      f'pr_{self.trainer.datamodule.idx2label[key]}': val
                      for key, val in zip(range(1, self.trainer.datamodule.num_classes), pr_per_label)
                  } | {
                      f'rc_{self.trainer.datamodule.idx2label[key]}': val
                      for key, val in zip(range(1, self.trainer.datamodule.num_classes), rc_per_label)
                  } | {
                      f'f1_{self.trainer.datamodule.idx2label[key]}': val
                      for key, val in zip(range(1, self.trainer.datamodule.num_classes), f1_per_label)
                  }

        metrics.update({
            'pr_micro': pr_micro,
            'rc_micro': rc_micro,
            'f1_micro': f1_micro,
            'pr_macro': pr_macro,
            'rc_macro': rc_macro,
            'f1_macro': f1_macro,
            'pr_weighted': pr_weighted,
            'rc_weighted': rc_weighted,
            'f1_weighted': f1_weighted,
        })
        return metrics

    def common_step(
            self,
            batch: tuple[list[int], list[list[int]], list[list[int]], BatchEncoding, torch.LongTensor]
    ) -> (List[torch.Tensor], List[torch.Tensor], float):
        doc_ids, first_token_pos, original_token_pos, batch_encoding, labels = batch
        batch_size = labels.shape[0]

        encoded = self.encoder(
            input_ids=batch_encoding.data['input_ids'],
            attention_mask=batch_encoding.data['attention_mask'],
            return_dict=True
        )['last_hidden_state']

        logits = F.softmax(self.head(encoded), dim=-1)

        attention_mask = batch_encoding.data['attention_mask']

        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_classes)
        active_labels = torch.where(
            active_loss,
            labels.view(-1),
            torch.tensor(self.loss.ignore_index).type_as(labels)
        )

        loss = self.loss(active_logits, active_labels)

        batch_preds = [logits[i, torch.nonzero(attention_mask[i])].squeeze() for i in range(batch_size)]
        batch_golds = [labels[i, torch.nonzero(attention_mask[i])].squeeze() for i in range(batch_size)]

        return batch_preds, batch_golds, loss

    def training_step(self, batch, batch_idx):
        preds, golds, loss = self.common_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(preds))
        return {
            'loss': loss,
            'preds': preds,
            'golds': golds
        }

    def validation_step(self, batch, batch_idx):
        preds, golds, loss = self.common_step(batch)
        self.log('val_loss', loss, batch_size=len(preds))
        return {
            'preds': preds,
            'golds': golds
        }

    # FIXME shouldn't it be validation_step_end instead? https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#validating-with-dataparallel
    def validation_epoch_end(self, validation_step_outputs: ValidationEpochOutputs) -> None:
        # TODO as a future feature we can aggregate results for each dataloader separately
        # flattening outputs from dataloaders if there are multiple
        if validation_step_outputs and isinstance(validation_step_outputs[0], list):
            outputs = [
                step_output
                for dataloader_output in validation_step_outputs
                for step_output in dataloader_output
            ]
        else:
            outputs = validation_step_outputs

        preds_logits = torch.cat([pred for step_output in outputs for pred in step_output['preds']])
        preds: torch.Tensor = torch.max(preds_logits, dim=1).indices
        golds: torch.Tensor = torch.cat([gold for step_output in outputs for gold in step_output['golds']])

        preds_numpy = preds.cpu().numpy()
        golds_numpy = golds.cpu().numpy()

        # rank_zero_info(str(preds_numpy))
        # rank_zero_info(str(golds_numpy))
        rank_zero_info(f'tp: {np.sum(np.where(golds_numpy != 0, preds_numpy == golds_numpy, False))}')
        rank_zero_info(f'nonzero count:, {np.count_nonzero(preds_numpy)}')
        rank_zero_info(f'preds_logits: {preds_logits[0]}')

        metrics = self._calculate_metrics(preds_numpy, golds_numpy)
        self.log_dict(metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        preds, golds, loss = self.common_step(batch)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, eps=1e-8)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.trainer.learning_rate)
        return optimizer

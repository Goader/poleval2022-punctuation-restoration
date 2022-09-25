from typing import Optional, Iterable, List, Dict, Tuple
from pathlib import Path
import json

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning.utilities import rank_zero_info
from transformers import PreTrainedTokenizer, AutoTokenizer

from omegaconf import DictConfig

from datatypes import Label, DatasetItem, PunctuationMark


class PunctuationRestorationDataModule(pl.LightningDataModule):
    PUNCTUATION_MARKS = ['.', ',', '?', '!', '-', '...']

    def __init__(
            self,
            cfg: DictConfig
    ):
        super().__init__()

        self.cfg = cfg

        self.tokenizer_name: str = cfg.model.encoder.tokenizer
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        self.train_batch_size: int = cfg.trainer.train_batch_size
        self.val_batch_size: int = cfg.trainer.val_batch_size
        self.test_batch_size: int = cfg.trainer.test_batch_size
        self.predict_batch_size: int = cfg.trainer.predict_batch_size

        self.train_dataset: Optional[Iterable[DatasetItem]] = None
        self.val_dataset: Optional[Iterable[DatasetItem]] = None
        self.test_dataset: Optional[Iterable[DatasetItem]] = None
        self.predict_dataset: Optional[Iterable[DatasetItem]] = None

        self.idx2label: List[Label] = ['O'] + self.PUNCTUATION_MARKS
        self.label2idx: Dict[Label, int] = {label: idx for idx, label in enumerate(self.idx2label)}
        self.num_classes: int = len(self.idx2label)

    def prepare_data(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        del tokenizer

    def _parse_tsv_input(self, filepath: Path) -> Iterable[DatasetItem]:
        pass

    # TODO max_seq_len
    def _parse_json_input(self, filepath: Path) -> Iterable[DatasetItem]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items: List[DatasetItem] = []
        for document in data[:100]:
            first_token_pos: List[int] = []
            token_ids: List[int] = []
            label_ids: List[int] = []
            for word in document['words']:
                if len(token_ids) >= self.cfg.trainer.max_seq_len-2:
                    break

                first_token_pos.append(len(token_ids))

                subtokens = self.tokenizer.tokenize(word['word'], )
                token_ids.extend(self.tokenizer.convert_tokens_to_ids(subtokens))

                label_ids.extend([self.label2idx.get(word['punctuation'], 0)] + [0] * (len(subtokens) - 1))

            token_ids = [self.tokenizer.cls_token_id] + token_ids[:self.cfg.trainer.max_seq_len-2] + [self.tokenizer.sep_token_id]
            label_ids = [0] + label_ids[:self.cfg.trainer.max_seq_len-2] + [0]

            items.append(DatasetItem(first_token_pos, token_ids, label_ids))

        return items

    # TODO other formats?

    def setup(self, stage: Optional[str] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if stage is None or stage == 'train':
            if self.cfg.data.train.endswith('.json'):
                self.train_dataset = self._parse_json_input(self.cfg.data.train)
            else:
                raise NotImplementedError('unknown format')

        if stage is None or stage == 'val':
            if self.cfg.data.val.endswith('.json'):
                self.val_dataset = self._parse_json_input(self.cfg.data.val)
            else:
                raise NotImplementedError('unknown format')

        if stage is None or stage == 'test':
            if self.cfg.data.test.endswith('.json'):
                self.test_dataset = self._parse_json_input(self.cfg.data.test)
            else:
                raise NotImplementedError('unknown format')

        if stage is None or stage == 'predict':
            if self.cfg.data.predict.endswith('.json'):
                self.predict_dataset = self._parse_json_input(self.cfg.data.predict)
            else:
                raise NotImplementedError('unknown format')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          collate_fn=self.collator,
                          num_workers=6)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          collate_fn=self.collator,
                          num_workers=3)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          collate_fn=self.collator,
                          num_workers=3)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.predict_dataset,
                          batch_size=self.predict_batch_size,
                          shuffle=False,
                          collate_fn=self.collator,
                          num_workers=3)

    def collator(self, batch: Iterable[DatasetItem]) -> Tuple[List[List[int]], List[torch.Tensor], List[torch.Tensor]]:
        first_token_pos = [item.first_token_pos for item in batch]
        token_ids = [item.token_ids for item in batch]
        label_ids = [torch.tensor(item.label_ids) for item in batch]

        padded_token_ids = [self.tokenizer.pad(
            {'input_ids': tokens},
            padding='longest',
            return_tensors='pt'
        ) for tokens in token_ids]

        return first_token_pos, padded_token_ids, label_ids

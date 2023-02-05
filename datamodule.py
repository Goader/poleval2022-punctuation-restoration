from typing import Optional, Iterable, List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from pytorch_lightning.utilities import rank_zero_info
from transformers import PreTrainedTokenizer, AutoTokenizer, BatchEncoding

import tqdm
from omegaconf import DictConfig

from datatypes import Label, PunctuationMark


@dataclass(frozen=True)
class DatasetItem:
    """
    :doc_id: - identifier of the original document
    :first_subtoken_pos: - mapping of the high-level token position to its first subtoken.
        Length must be equal to the number of high-level tokens (not subtokens).
    :original_token_pos: - mapping of the subtoken position to the high-level token index.
        Length must be equal to the number of subtokens.
    :token_ids: - list of subtoken indices originating from the tokenizer (indices from the vocabulary).
        Length must be equal to the number of subtokens.
    :label_ids: - list of labels indices that correspond to the original tokens (indices from the label2idx mapping).
        Length must be equal to the number of high-level tokens (each token has its label).
    """

    doc_id: int
    first_subtoken_pos: list[int]
    original_token_pos: list[int]
    token_ids: list[int]
    label_ids: list[int]

    def __post_init__(self):
        assert len(self.first_subtoken_pos) == len(self.label_ids), \
            'the number of tokens does not match the number of labels'
        assert len(self.first_subtoken_pos) <= len(self.token_ids), \
            'there cannot be more words (tokens), than subtokens'
        assert len(self.original_token_pos) == len(self.token_ids), \
            'the length of original token mapping must be the same as the number of subtokens'
        assert all(pos < len(self.token_ids) for pos in self.first_subtoken_pos), \
            'word\'s first token position cannot exceed the number of subtokens'


# TODO update comments regarding the variables' names change
# FIXME check what if we are starting with first token having only one subtoken, and the next having window_size
def windowed_tokens(
        first_subtoken_pos: list[int],
        original_token_pos: list[int],
        window_size: int,
        stride: float = 1.0
) -> list[tuple[int, int]]:
    """
    Splits the sequence of subtokens into windows of tokens, but leaves tokens undivided.
    This way only the token boundaries can be the boundaries of the windows themselves.

    :param first_subtoken_pos: indices of the start of each token
    :param original_token_pos: mapping of each subtoken id to its token id
    :param window_size: desired window size (the real may vary depending on the tokens position)
    :param stride: step is calculated by multiplying the resulting window size and stride
    :return: list of tuples, where each tuple contains
             a start index and an end index of the high-level token (left inclusive, right exclusive), not subtokens
    """

    start_token_pos = 0
    # the desired subtoken position to end the window in, but probably this position will be between the
    # two subtokens that belong to the same token, then we cannot end the window there, hence "desired"
    desired_end = window_size

    # if the desired end lands directly inside the last token of the whole sequence, then the code below will not notice
    # and yield a window, which includes it (for loop ends, and we proceed to `else` case).
    # This window will be too big (the last token is out of bounds), thus we need to catch this situation.
    # Adding the "ghost" token at the end will let us tackle the problem described above, the last token will land in
    # the next window, but if the desired end does not land inside the last token, it will simply add one iteration more
    # and then end as normal going to `else` case and just taking everything from the start token without "ghost" one
    first_subtoken_pos = first_subtoken_pos + [len(original_token_pos)]

    windows = []
    while True:
        for token_pos_offset, subtoken_pos in enumerate(first_subtoken_pos[start_token_pos:]):
            token_id = start_token_pos + token_pos_offset

            # if current token's position is already outside the window,
            # then we do not want it or even the previous token,
            # because despite the fact that the previous one started in the window boundaries ("if" was not satisfied),
            # it ended out of them (the start of the next token is already out), and thus we cannot take it
            if subtoken_pos > desired_end:

                # if the previous token is the first token in the window, then it is too long, we cannot divide it
                if token_id - 1 == start_token_pos:
                    raise ValueError(f'{start_token_pos}th token has more subtokens '
                                     f'than the window can have (max is {window_size})')

                windows.append((start_token_pos, token_id-1))

                # real window size is the number of subtokens which found themselves in the window
                real_window_size = first_subtoken_pos[token_id-1] - first_subtoken_pos[start_token_pos]

                desired_start = int(first_subtoken_pos[start_token_pos] + stride * real_window_size)

                # there are two different scenarios:
                #  - stride=1.0 or any case where `desired_start` lands exactly between the tokens:
                #       `desired_start` will be equal to the end position of the last token in the window.
                #       Then we want the `start_token_pos` to be the token, which starts in `desired_start`,
                #       thus taking the previous token id and adding one will work
                #  - stride<=1.0 or any case where `desired_start` lands inside a token:
                #       Since we've already taken this token (stride <= 1.0 and we divide strictly on token boundaries)
                #       then we can skip it, and we do skip it, because there might be the situation, when the first
                #       token is so long, that `desired_start` lands inside it, and then taking it will mean we start
                #       in the same place -> endless loop. We skip it simply by taking the token id of the previous
                #       subtoken, which is the id of the current token (we landed inside it) and take the next token id
                start_token_pos = original_token_pos[desired_start - 1] + 1
                desired_end = first_subtoken_pos[start_token_pos] + window_size

                break

        # the loop above will not get intercepted by `break` only if we finish the loop,
        # and that means we have finished the sequence, and all the tokens are inside its boundaries (comments above),
        # therefore we can say it is the last window, so we simply take everything from the start token to the very end
        else:
            windows.append((start_token_pos, len(first_subtoken_pos) - 1))  # omitting the "ghost" token we added
            break  # breaking the `while True` loop :)

    return windows


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

        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

    def prepare_data(self) -> None:
        pass

    def _parse_tsv_input(self, filepath: Path) -> Iterable[DatasetItem]:
        pass

    # TODO max_seq_len
    def _parse_json_input(self, filepath: Path) -> Iterable[DatasetItem]:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items: List[DatasetItem] = []
        for document_id, document in enumerate(tqdm.tqdm(data, desc='preprocessing documents')):
            first_subtoken_pos: list[int] = []
            original_token_ids: list[int] = []
            token_ids: list[int] = []
            label_ids: list[int] = []
            for original_token_id, word in enumerate(document['words']):
                subtokens = self.tokenizer.tokenize(word['word'], )  # todo limit the token length by `max_seq_len - ...`

                first_subtoken_pos.append(len(token_ids))
                original_token_ids.extend([original_token_id] * len(subtokens))

                token_ids.extend(self.tokenizer.convert_tokens_to_ids(subtokens))
                label_ids.append(self.label2idx.get(word['punctuation'], 0))

            items.append(DatasetItem(document_id, first_subtoken_pos, original_token_ids, token_ids, label_ids))

        return items

    def split_into_windows(self, documents: Iterable[DatasetItem]) -> Iterable[Iterable[DatasetItem]]:
        window_size: int = self.cfg.trainer.max_seq_len - 2
        stride: float = self.cfg.trainer.stride

        splitted_documents: list[list[DatasetItem]] = []
        for item in tqdm.tqdm(documents, desc='splitting documents using sliding window'):
            windows = windowed_tokens(item.first_subtoken_pos, item.original_token_pos, window_size, stride)

            document_windows: list[DatasetItem] = []
            for start, end in windows:
                # in case `end` is indicating the last token, we are out of bounds,
                # so we simply set the `subtoken_end` to the last subtoken
                subtoken_start = item.first_subtoken_pos[start]
                subtoken_end = item.first_subtoken_pos[end] \
                    if end < len(item.first_subtoken_pos) else len(item.token_ids)

                window_item = DatasetItem(
                    item.doc_id,
                    first_subtoken_pos=[
                        subtoken_pos - subtoken_start
                        for subtoken_pos in item.first_subtoken_pos[start:end]
                    ],
                    original_token_pos=[
                        token_pos - start
                        for token_pos in item.original_token_pos[subtoken_start:subtoken_end]
                    ],
                    token_ids=item.token_ids[subtoken_start:subtoken_end],
                    label_ids=item.label_ids[start:end]
                )

                document_windows.append(window_item)

            splitted_documents.append(document_windows)

        return splitted_documents

    def setup(self, stage: Optional[str] = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        if stage is None or stage == 'train':
            if self.cfg.data.train.endswith('.json'):
                documents = self._parse_json_input(self.cfg.data.train)
                windowed_documents = self.split_into_windows(documents)
                self.train_dataset = [
                    window
                    for document in windowed_documents
                    for window in document
                ]
            else:
                raise NotImplementedError('unknown format')

        if stage is None or stage == 'val':
            if self.cfg.data.val.endswith('.json'):
                documents = self._parse_json_input(self.cfg.data.val)
                windowed_documents = self.split_into_windows(documents)
                self.val_dataset = [
                    window
                    for document in windowed_documents
                    for window in document
                ]
            else:
                raise NotImplementedError('unknown format')

        if stage is None or stage == 'test':
            if self.cfg.data.test.endswith('.json'):
                documents = self._parse_json_input(self.cfg.data.test)
                windowed_documents = self.split_into_windows(documents)
                self.test_dataset = [
                    window
                    for document in windowed_documents
                    for window in document
                ]
            else:
                raise NotImplementedError('unknown format')

        if stage is None or stage == 'predict':
            if self.cfg.data.predict.endswith('.json'):
                documents = self._parse_json_input(self.cfg.data.predict)
                windowed_documents = self.split_into_windows(documents)
                self.predict_dataset = [
                    window
                    for document in windowed_documents
                    for window in document
                ]
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

    def collator(self, batch: Iterable[DatasetItem]) \
            -> tuple[list[int], list[list[int]], list[list[int]], BatchEncoding, torch.LongTensor]:

        doc_ids = [item.doc_id for item in batch]
        first_subtoken_pos = [item.first_subtoken_pos for item in batch]
        original_token_pos = [item.original_token_pos for item in batch]

        token_ids = [
            [self.tokenizer.cls_token_id] + item.token_ids + [self.tokenizer.sep_token_id]
            for item in batch
        ]  # fixme is this cls and sep token ok?
        batch_encoding = self.tokenizer.pad(
            {'input_ids': token_ids},
            padding='longest',
            return_tensors='pt',
            return_attention_mask=True
        )

        pad_len = batch_encoding.data['input_ids'].shape[1]  # fixme safer way of getting seq len?
        padded_label_ids = torch.tensor([
            item.label_ids + [0] * (pad_len - len(item.label_ids))
            for item in batch
        ], dtype=torch.long)

        return doc_ids, first_subtoken_pos, original_token_pos, batch_encoding, padded_label_ids

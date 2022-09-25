from typing import Literal, Union, List, Dict, Any
from dataclasses import dataclass

import torch


ZeroClass = Literal['O']
PunctuationMark = Literal['.', ',', '?', '!', '-', '...']
Label = Union[ZeroClass, PunctuationMark]

ValidationEpochOutputs = Union[
    List[
        Union[torch.Tensor, Dict[str, Any]]
    ],

    List[List[
        Union[torch.Tensor, Dict[str, Any]]
    ]]
]


@dataclass(frozen=True)
class DatasetItem:
    first_token_pos: List[int]
    token_ids: List[int]
    label_ids: List[int]

    def __post_init__(self):
        assert len(self.token_ids) == len(self.label_ids), \
            'the number of tokens does not match the number of labels'
        assert len(self.first_token_pos) < len(self.token_ids), \
            'there cannot be more words, than subtokens'
        assert all(pos < len(self.token_ids) for pos in self.first_token_pos), \
            'word\'s first token position cannot exceed the number of subtokens'

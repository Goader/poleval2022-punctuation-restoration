from typing import Literal, Union, List, Dict, Any

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

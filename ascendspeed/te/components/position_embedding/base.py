from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Type, TypeVar

import torch.nn as nn

Self = TypeVar("Self", bound="PositionEmbedding")


@dataclass
class PositionEmbeddingConfig:
    name: str
    dim_model: int
    seq_len: int


class PositionEmbedding(nn.Module, metaclass=ABCMeta):
    r""" The base PositionEmbedding abstract class """
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def from_config(cls: Type[Self], config: PositionEmbeddingConfig) -> Self:
        # Generate the class inputs from the config
        fields = asdict(config)

        # Skip all Nones so that default values are used
        fields = {k: v for k, v in fields.items() if v is not None}
        return cls(**fields)

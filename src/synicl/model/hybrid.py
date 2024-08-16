from dataclasses import dataclass
from typing import Any

import torch

from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutput
from transformers import PretrainedConfig
from yaml import YAMLObject

# from .base import TrainableModel
from ..core import AbstractionError

@dataclass
class HybridLayer(nn.Module):

    def setup(self, layer_idx: int) -> None:
        raise AbstractionError(f"Abstract Class HybridLayer has no `.setup()`!")
    
    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.net(hidden_state)

class ResidualMarker(HybridLayer, YAMLObject):
    """
    A dummy layer to mark where 
    residual connections should be made
    """

    yaml_tag = u"!Layer:Residual"

    def setup(self, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.net = nn.Identity()

@dataclass
class HybridBackbone(nn.Module):
    
    layers: list[HybridLayer]

    def setup(self):
        nn.Module.__init__(self)
        for idx, l in enumerate(self.layers):
            l.setup(layer_idx=idx)

        self._layers = nn.ModuleList(self.layers)
    
    def forward(self, inputs_embeds: Tensor) -> BaseModelOutput:
        hidden_state = inputs_embeds
        residual = 0

        for layer in self._layers:
            layer = layer.to(hidden_state.device)
            hidden_state: Tensor = layer(hidden_state)

            if isinstance(layer, ResidualMarker):
                hidden_state = residual + hidden_state
                residual = hidden_state

        return BaseModelOutput(
            last_hidden_state=hidden_state # pyright: ignore[reportArgumentType]
        )

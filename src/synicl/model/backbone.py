from dataclasses import dataclass
from typing import Any

from yaml import YAMLObject
from torch import nn, Tensor

from ..core import Curriculum
from .base import TrainableModel

@dataclass(unsafe_hash=True)
class BackboneModel(TrainableModel, YAMLObject):

    yaml_tag = u'!BackboneModel'

    backbone: nn.Module
    in_dim: int
    hidden_dim: int = 128
    out_dim: int = 1

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "in_dim" : self.in_dim,
            "hidden_dim" : self.hidden_dim,
            "out_dim" : self.out_dim,
        } | ({
            "config" : self.backbone.config.to_dict()
        } if hasattr(self.backbone, "config") else { })

    def setup(self):
        if getattr(self, "is_setup", False):
            return

        nn.Module.__init__(self)
        in_dim = Curriculum.wrap(self.in_dim, int).max
        hidden_dim = Curriculum.wrap(self.hidden_dim, int).max
        out_dim = Curriculum.wrap(self.out_dim, int).max

        self._read_in = nn.Linear(in_dim, hidden_dim)
        self._read_out = nn.Linear(hidden_dim, out_dim)

        self.is_setup = True

    def forward(self, xs: Tensor, ys: Tensor):
        self.setup()
        self.backbone.to(xs.device)

        zs = self.interleave(xs, ys)
        embeds = self._read_in(zs)
        output = self.backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)

        return prediction[:, ::2] # predict only on xs

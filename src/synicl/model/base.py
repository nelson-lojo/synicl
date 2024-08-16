from dataclasses import dataclass, field
from typing import Any

import torch
from yaml import YAMLObject
from torch import nn, Tensor

from ..core import Curriculum

@dataclass
class ContextModel:

    name: str = field(default="Unknown_ContextModel", kw_only=True)

    def __str__(self):
        return self.name

    def get_dims(self, xs: Tensor, ys: Tensor) -> tuple[int, int, tuple[int, int]]:
        x_bs, x_seq, x_dim = xs.shape
        _, _, y_dim = ys.shape

        return (x_bs, x_seq, (x_dim, y_dim))

    def evaluate(self, xs: Tensor, ys: Tensor) -> Tensor:
        """
        Translate from a sequence of x,y pairs to predicted y 
        values for each presented x value. `xs` must be the 
        same length as or exactly one longer than `ys`
        """
        raise NotImplementedError(f"Abstract class ContextModel does not implement `.evaluate()`!")

    # Helper for .forward
    def interleave(self, xs: Tensor, ys: Tensor) -> Tensor:
        # code adapted from Garg et. al.
        """Interleaves the x's and the y's into a single sequence with shape (batch_size, 2*num_points, x_dim)"""
        bsize, points, (x_dim, y_dim) = self.get_dims(xs, ys)

        ys_wide = torch.cat(
            (
                ys.view(bsize, points, y_dim),
                torch.zeros(bsize, points, x_dim - y_dim, device=ys.device),
            ),
            dim=2,
        )
        zs = torch.stack((xs, ys_wide), dim=2)
        zs = zs.view(bsize, 2 * points, x_dim)
        return zs

    def stack(self, xs: Tensor, ys: Tensor, ctx_len: int = 1) -> Tensor:
        """Stacks the x's and the y's into a single sequence with shape (batch_size, num_points, x_dim + ctx_len * (y_dim + x_dim). Relies on `self.context_len`"""
        bsize, points, (x_dim, y_dim) = self.get_dims(xs, ys)

        xy_seq = torch.cat(
            (xs, 
             ys.view(ys.shape + (1,))), 
            dim=2
        )

        contexted = [
            torch.cat((torch.zeros(bsize, i, x_dim+y_dim), xy_seq[:, :-i,:]), dim=1)
            for i in range(1, ctx_len + 1)
        ]

        return torch.cat(contexted + [xs], dim=-1) # returns (b_size, seq_len, x_dim + ctx_len * (x_dim + y_dim))


@dataclass
class Baseline(ContextModel): ...

@dataclass
class TrainableModel(ContextModel, nn.Module):

    def setup(self):
        pass

    @property
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError(f"Abstract TrainableModel does not have metadata!")

    def forward(self, xs: Tensor, ys: Tensor) -> Tensor:
        raise NotImplementedError(f"Abstract TrainableModel does not implement .forward()!")
    
    def evaluate(self, xs: Tensor, ys: Tensor) -> Tensor:
        return self(xs, ys)

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
            "config" : self.backbone.config.to_dict()
        }

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

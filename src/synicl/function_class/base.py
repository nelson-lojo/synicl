# pyright: reportMissingSuperCall=information
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, Size
from yaml import YAMLObject

from ..core import (
    XDistribution,
    Curriculum,
)

@dataclass
class FunctionClass(YAMLObject):

    x_distribution: XDistribution
    batch_size: int
    sequence_length: int | Curriculum[int]
    y_dim: int

    def sample_at_step(self, step_num: int) -> tuple[Tensor, Tensor]:
        if not isinstance(self.sequence_length, Curriculum):
            self.sequence_length = Curriculum.wrap(self.sequence_length, int)

        x_batch = self.x_distribution.sample_at_step(
            step_num, 
            Size([
                self.batch_size,
                self.sequence_length.at_step(step_num)
            ])
        )

        params = self.sample_params(step_num)
        if isinstance(params, list):
            y_batch: torch.Tensor = self.evaluate(x_batch, *params, step_num=step_num)
        else:
            y_batch: torch.Tensor = self.evaluate(x_batch, params, step_num=step_num)

        if torch.cuda.is_available():
            return x_batch.cuda(), y_batch.cuda() 
        else:
            return x_batch, y_batch

    def sample_params(self, step_num: int) -> Tensor | list[Tensor]:
        """Randomly sample parameters"""
        raise NotImplementedError(f"Abstract class FunctionClass cannot sample parameters!")

    def evaluate(self, x_batch: Tensor, *params: torch.Tensor, step_num: int) -> Tensor:
        """Produce a Tensor of shape (batch_size, sequence_length, y_dim) given a Tensor of shape (batch_size, sequence_length, x_dim)"""
        raise NotImplementedError(f"Abstract class FunctionClass does not implement `.evaluate(xs)`!")

@dataclass
class ModifiedFunctionClass(YAMLObject):
    
    inner_function_class: FunctionClass

    #### BEGIN: Delegations ####

    @property
    def x_distribution(self):
        return self.inner_function_class.x_distribution

    @property
    def batch_size(self) -> int:
        return self.inner_function_class.batch_size

    @property
    def sequence_length(self) -> Curriculum[int]:
        return cast(Curriculum[int], self.inner_function_class.sequence_length)

    @property
    def y_dim(self) -> int:
        return self.inner_function_class.y_dim
    
    def sample_params(self, step_num: int) -> Tensor | list[Tensor]:
        return self.inner_function_class.sample_params(step_num)

    def evaluate(self, x_batch: Tensor, *params: torch.Tensor, step_num: int) -> Tensor:
        return self.inner_function_class.evaluate(x_batch, *params, step_num=step_num)

    #### END: Delegations ####

    def modify_x_pre_eval(self, x_batch: Tensor) -> Tensor:
        return x_batch

    def modify_x_post_eval(self, x_batch: Tensor) -> Tensor:
        return x_batch

    def modify_y(self, y_batch: Tensor) -> Tensor:
        return y_batch

    def sample_at_step(self, step_num: int) -> tuple[Tensor, Tensor]:
        x_batch = self.x_distribution.sample_at_step(
            step_num, 
            Size([
                self.batch_size,
                self.sequence_length.at_step(step_num)
            ])
        )
        x_batch = self.modify_x_pre_eval(x_batch)

        params = self.sample_params(step_num)
        if isinstance(params, list):
            y_batch: torch.Tensor = self.evaluate(x_batch, *params, step_num=step_num)
        else:
            y_batch: torch.Tensor = self.evaluate(x_batch, params, step_num=step_num)

        if torch.cuda.is_available():
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        return (
            self.modify_x_post_eval(x_batch),
            self.modify_y(y_batch)
        )

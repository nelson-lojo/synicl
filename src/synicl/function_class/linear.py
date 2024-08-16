from dataclasses import dataclass

import torch
from torch import Tensor
from yaml import YAMLObject

from .base import FunctionClass

@dataclass
class LinearRegression(FunctionClass, YAMLObject):

    yaml_tag = u'!LinearRegression'

    def sample_params(self, step_num: int) -> Tensor | list[Tensor]:
        """Produce a set of sampled parameters"""
        
        param_dist_shape = torch.Size([
            self.batch_size, 1, self.y_dim, self.x_distribution.max_x_dim
        ])

        params = torch.normal( torch.zeros(param_dist_shape), 
                               torch.ones(param_dist_shape)   )

        return params

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor, step_num: int) -> torch.Tensor:
        weights, *_ = params
        partial_sums = torch.bmm(weights.squeeze(-2), x_batch.permute(0, 2, 1))
        full_sums = torch.sum(partial_sums, dim=-2, keepdim=True)
        y_batch = full_sums.transpose(-2, -1)

        return y_batch

@dataclass
class SparseLinearRegression(LinearRegression, YAMLObject):

    yaml_tag = u'!SparseLinearRegression'

    sparsity: int = 3

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor, step_num: int) -> torch.Tensor:
        weights, *_ = params

        mask = torch.ones(weights.shape).bool()

        for batch_entry in range(weights.shape[0]):

            dimensions_to_keep: torch.Tensor = torch.randperm(
                    self.x_distribution.x_curriculum.at_step(step_num)
                )[:self.sparsity]

            mask[batch_entry, ..., dimensions_to_keep] = False

        weights[mask] = 0

        return super().evaluate(x_batch, weights, step_num=step_num)

@dataclass
class LinearClassification(LinearRegression, YAMLObject):

    yaml_tag = u'!LinearClassification'

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor, step_num: int):
        y_batch = super().evaluate(x_batch, *params, step_num=step_num)
        return y_batch.sign()

@dataclass
class QuadraticRegression(LinearRegression, YAMLObject):

    yaml_tag = u'!QuadraticRegression'

    def evaluate(self, x_batch: torch.Tensor, *params: torch.Tensor, step_num: int) -> torch.Tensor:
        return super().evaluate(x_batch**2, *params, step_num=step_num)

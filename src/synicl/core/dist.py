from dataclasses import dataclass
from typing import Any

import torch
import torch.distributions as D
from torch import Size, Tensor
from yaml import YAMLObject

from .curriculum import Curriculum
# from .exceptions import AbstractionError

@dataclass
class XDistribution(YAMLObject):

    yaml_tag = u'!XDistribution'

    @property
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError(f"Metadata not yet implemented")
    
    inner_distribution: type[D.Distribution]
    x_dimension: int | Curriculum[int]
    
    @property
    def max_x_dim(self) -> int:
        return Curriculum(self.x_dimension).max
    
    @property
    def x_curriculum(self) -> Curriculum[int]:
        return Curriculum[int](self.x_dimension)

    def _populate_inner_dist(self):
        if hasattr(self, "_dist"):
            return

        x_dim: Curriculum[int] = self.x_curriculum
        match self.inner_distribution:
            case D.MultivariateNormal:
                self._dist = D.MultivariateNormal(
                    loc=torch.zeros(x_dim.max),
                    covariance_matrix=torch.eye(x_dim.max)
                )
            case D.Uniform:
                self._dist = D.Uniform(
                    -torch.ones(x_dim.max),
                    torch.ones(x_dim.max)
                )
            case _:
                raise NotImplementedError(f"Distribution {self.inner_distribution} not yet supported!")
        

    def sample_at_step(self, step_num: int, sample_shape: Size) -> Tensor:
        self._populate_inner_dist()

        sample = self._dist.sample(sample_shape=sample_shape)
        current_dimension = Curriculum(self.x_dimension).at_step(step_num)
        sample[..., current_dimension:] = 0
        return sample
        

from dataclasses import dataclass

import torch
import torch.distributions as D
from torch import Size, Tensor
from yaml import YAMLObject

from .curriculum import Curriculum

@dataclass
class XDistribution(YAMLObject):

    yaml_tag = u'!XDistribution'
    
    inner_distribution: type[D.Distribution]
    x_dimension: int | Curriculum[int]
    
    @property
    def max_x_dim(self) -> int:
        return Curriculum.wrap(self.x_dimension, int).max
    
    @property
    def x_curriculum(self) -> Curriculum[int]:
        return Curriculum.wrap(self.x_dimension, int)

    def _setup(self):
        if hasattr(self, "is_setup"):
            return

        self._x_dim: Curriculum[int] = self.x_curriculum
        match self.inner_distribution:
            case D.MultivariateNormal:
                self._dist = D.MultivariateNormal(
                    loc=torch.zeros(self._x_dim.max),
                    covariance_matrix=torch.eye(self._x_dim.max)
                )
            case D.Uniform:
                self._dist = D.Uniform(
                    -torch.ones(self._x_dim.max),
                    torch.ones(self._x_dim.max)
                )
            case _:
                raise NotImplementedError(f"Distribution {self.inner_distribution} not yet supported!")
        
        self.is_setup = True

    def _sample_at_step(self, step_num: int, sample_shape: Size) -> Tensor:
        return self._dist.sample(sample_shape=sample_shape)

    def sample_at_step(self, step_num: int, sample_shape: Size) -> Tensor:
        self._setup()

        sample = self._sample_at_step(step_num, sample_shape)
        current_dimension = self._x_dim.at_step(step_num)
        sample[..., current_dimension:] = 0
        return sample
        

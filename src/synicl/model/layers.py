from typing import Any
from dataclasses import dataclass
import inspect

import torch
from torch import nn, Tensor
from transformers import PretrainedConfig

from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaAttention,
    LlamaRMSNorm,
)

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP as Gpt2Mlp

from yaml import YAMLObject

from ..core import Curriculum
from .hybrid import HybridLayer

@dataclass
class InstantiateLayer(HybridLayer, YAMLObject):

    yaml_tag = u'!Layer:Instantiate'

    config: PretrainedConfig
    layer_type: type[nn.Module]
    
    def setup(self, layer_idx: int) -> None:
        nn.Module.__init__(self)
        sig = inspect.signature(self.layer_type)

        kwargs = { }
        if "layer_idx" in sig.parameters:
            kwargs |= { "layer_idx": layer_idx }

        if "config" in sig.parameters:
            self.net = self.layer_type(config=self.config, **kwargs)
        else:
            self.net = self.layer_type(self.config, **kwargs)

    def forward(self, hidden_state: Tensor) -> Tensor:
        out = self.net(hidden_state)

        if isinstance(out, (tuple, )): # collect only attention outputs for attn layers
            return out[0]

        return out

@dataclass
class AbsolutePositionalEmbedding(HybridLayer, YAMLObject):

    yaml_tag = u"!Layer:AbsolutePositionalEmbedding"

    max_num_positions: int
    hidden_dim: int

    def setup(self, layer_idx: int):
        nn.Module.__init__(self)
        self.positions = torch.arange(0, self.max_num_positions, dtype=torch.int)
        self.embed = nn.Embedding(self.max_num_positions, self.hidden_dim)

    def forward(self, hidden_state: Tensor) -> Tensor:
        *_, seq_len, _ = hidden_state.shape
        pos = self.positions.to(device=hidden_state.device)
        embeddings = self.embed(pos)[:seq_len]
        return hidden_state + embeddings

@dataclass
class LlamaAttn(HybridLayer, YAMLObject):

    yaml_tag = u"!Layer:LlamaAttention"

    class _RotaryEmbeddingStub(LlamaRotaryEmbedding):
        def __init__(self, *args: Any, enable: bool = False, **kwargs: Any):
            super().__init__(*args, **kwargs)

            self.enable = enable

        def forward(self, x: Tensor, position_ids: Tensor):
            cos, sin = super().forward(x, position_ids)

            if self.enable:
                return cos, sin
            return torch.ones_like(cos), torch.zeros_like(sin)

    config: PretrainedConfig
    use_rope: bool
    b_size: int
    seq_len: int | Curriculum[int]

    def setup(self, layer_idx: int):
        nn.Module.__init__(self)
        self.net = LlamaAttention(config=self.config, layer_idx=layer_idx) # pyright: ignore[reportArgumentType]
        self.net.rotary_emb = LlamaAttn._RotaryEmbeddingStub(
            config=self.config,
            device=None,
            enable=self.use_rope
        )
        seq_len: int = Curriculum.wrap(self.seq_len, int).max
        self.attention_mask = torch.triu(
            torch.full(
                (
                    self.b_size, self.config.num_attention_heads,
                    seq_len, seq_len
                ),
                fill_value=torch.finfo(torch.float32).min,
            ),
            diagonal=1
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        out = self.net(
            hidden_state,
            position_ids=torch.arange(
                0, hidden_state.shape[1],
                device=hidden_state.device
            ).unsqueeze(0),
            attention_mask=self.attention_mask
        )

        if isinstance(out, (tuple, )): # collect only attention outputs for attn layers
            return out[0]

        return out

@dataclass
class RMSNorm(HybridLayer, YAMLObject):

    yaml_tag = u'!Layer:RMSNorm'

    hidden_dim: int
    epsilon: float

    def setup(self, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.net = LlamaRMSNorm(self.hidden_dim, eps=self.epsilon)

@dataclass
class LayerNorm(HybridLayer, YAMLObject):

    yaml_tag = u'!Layer:LayerNorm'

    hidden_dim: int
    epsilon: float

    def setup(self, layer_idx: int) -> None:
        nn.Module.__init__(self)
        self.net = nn.LayerNorm(self.hidden_dim, self.epsilon)

class GPT2MLP(HybridLayer, YAMLObject):
    
    yaml_tag = u'!Layer:GPT2MLP'

    config: GPT2Config

    def setup(self, layer_idx: int) -> None:
        nn.Module.__init__(self)

        in_dim = self.config.n_inner if self.config.n_inner is not None else 4 * self.config.hidden_size
        self.net = Gpt2Mlp(intermediate_size=in_dim, config=self.config)

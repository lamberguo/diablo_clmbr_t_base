"""
DiaBlo: Diagonal Block Linear module for parameter-efficient fine-tuning.
"""

import math

import torch
import torch.nn as nn


class BlockLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int,
        bias: bool = False,
        drop_out: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

        self.block_size_in = math.ceil(in_features / num_blocks)
        self.block_size_out = math.ceil(out_features / num_blocks)
        n1 = math.ceil(in_features / self.block_size_in)
        n2 = math.ceil(out_features / self.block_size_out)
        self.num_blocks = max(n1, n2)

        self.register_parameter(
            "block_A",
            nn.Parameter(torch.zeros(self.num_blocks, self.block_size_in, self.block_size_out)),
        )
        self.in_diff = self.block_size_in * self.num_blocks - in_features
        self.dropout = nn.Dropout(drop_out)

    def block_forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 3:
            prefix = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            y = self.block_forward(x)
            return y.reshape(*prefix, self.out_features)
        if x.ndim == 2:
            if self.in_diff > 0:
                y = torch.zeros(x.shape[0], x.shape[1] + self.in_diff, device=x.device, dtype=x.dtype)
                y[:, : x.shape[1]] = x
                x = y
            x = x.view(x.size(0), self.num_blocks, self.block_size_in)
            outshape = (x.size(0), self.out_features)
            einsum_str = "bij,ijk->bik"
        elif x.ndim == 3:
            if self.in_diff > 0:
                y = torch.zeros(
                    x.shape[0], x.shape[1], x.shape[2] + self.in_diff, device=x.device, dtype=x.dtype
                )
                y[:, :, : x.shape[2]] = x
                x = y
            x = x.view(x.size(0), x.size(1), self.num_blocks, self.block_size_in)
            outshape = (x.size(0), x.size(1), self.out_features)
            einsum_str = "blij,ijk->blik"
        else:
            raise ValueError(f"Input tensor must have 2 or 3 dimensions, got {x.ndim}.")

        result = torch.einsum(einsum_str, x, self.block_A)
        result = result.flatten(start_dim=-2)[..., : self.out_features].reshape(outshape)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        result = result + self.block_forward(self.dropout(x))
        return result


def replace_blocklinear_with_linear(model: nn.Module) -> None:
    """Merge BlockLinear adapters back into standard nn.Linear layers."""
    for name, module in model.named_children():
        if isinstance(module, BlockLinear):
            d1, d2 = module.block_size_in, module.block_size_out
            n = module.num_blocks
            d = torch.zeros(n, d1, n, d2, device=module.linear.weight.device, dtype=module.linear.weight.dtype)
            inds = torch.arange(n)
            d[inds, :, inds, :] = module.block_A.data
            d = torch.reshape(d, (n * d1, n * d2))[: module.in_features, : module.out_features]
            module.linear.weight.data = module.linear.weight.data + d.T
            setattr(model, name, module.linear)
        else:
            replace_blocklinear_with_linear(module)


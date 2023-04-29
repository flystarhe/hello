# pip install -U fvcore
import torch
from fvcore.nn import (FlopCountAnalysis, flop_count_table,
                       parameter_count_table)


def print_parameter_count_table(model, max_depth=1):
    """Format the parameter count of the model in a nice table.

    Args:
        model: A torch module.
        max_depth (int): The max depth of submodules to include in the table. Defaults to 1.
    """
    print(parameter_count_table(model, max_depth))


def print_flop_count_table(model, input_shape, max_depth=1):
    """Format the per-module parameters and flops of a model in a table.

    Args:
        model: A torch module.
        input_shape: The input shape to the model for analysis.
        max_depth (int): The max depth of submodules to include in the table. Defaults to 1.
    """
    if len(input_shape) == 2:
        input_shape = (1, 3) + tuple(input_shape)
    assert len(input_shape) == 4, "such as `(h,w)` or `(n,c,h,w)`"

    inputs = (torch.rand(input_shape),)

    flops = FlopCountAnalysis(model, inputs)

    print(flop_count_table(flops, max_depth))

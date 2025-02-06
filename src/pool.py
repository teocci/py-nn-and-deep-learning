"""
Created by Teocci.
Author: teocci@yandex.com
Date: 2025-2월-06
"""
import warnings
from typing import Optional, Tuple, Union

from pytensor.tensor import as_tensor_variable, Variable
from pytensor.tensor import pad as tensor_pad
from pytensor.tensor import shape, max, sum, arange
from pytensor.tensor.math import floor


def pool_2d(
        input,
        ws: Optional[Union[Tuple[int, int], Variable]] = None,
        ignore_border: Optional[bool] = None,
        stride: Optional[Union[Tuple[int, int], Variable]] = None,
        pad: Tuple[int, int] = (0, 0),
        mode: str = "max",
        ds: Optional[Union[Tuple[int, int], Variable]] = None,
        st: Optional[Union[Tuple[int, int], Variable]] = None,
        padding: Optional[Tuple[int, int]] = None,
) -> Variable:
    """
    Downscale the input by a specified factor using pooling.

    Parameters
    ----------
    input : TensorLike
        Input images. Pooling will be done over the last two dimensions.
    ws : tuple of length 2 or Variable, optional
        Factor by which to downscale (vertical, horizontal).
        (2, 2) will halve the image in each dimension.
    ignore_border : bool, optional
        When True, (5, 5) input with ws=(2, 2) will generate a (2, 2) output.
        (3, 3) otherwise.
    stride : tuple of two ints or Variable, optional
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If None, it is considered equal to `ws`.
    pad : tuple of two ints, optional
        Padding size for top/bottom and left/right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}, default='max'
        Operation executed on each window.
    ds : tuple of length 2 or Variable, optional
        Deprecated, use `ws` instead.
    st : tuple of two ints or Variable, optional
        Deprecated, use `stride` instead.
    padding : tuple of two ints, optional
        Deprecated, use `pad` instead.

    Returns
    -------
    Variable
        The pooled output tensor.
    """
    # Handle deprecated parameters
    if ds is not None:
        if ws is not None:
            raise ValueError("Cannot provide both 'ws' and 'ds'. Use 'ws' only.")
        warnings.warn("'ds' is deprecated; use 'ws' instead.", DeprecationWarning)
        ws = ds
    elif ds is None and ws is None:
        raise ValueError("Must provide a tuple value for the window size ('ws').")

    if st is not None:
        if stride is not None:
            raise ValueError("Cannot provide both 'st' and 'stride'. Use 'stride' only.")
        warnings.warn("'st' is deprecated; use 'stride' instead.", DeprecationWarning)
        stride = st

    if padding is not None:
        if pad != (0, 0):
            raise ValueError("Cannot provide both 'padding' and 'pad'. Use 'pad' only.")
        warnings.warn("'padding' is deprecated; use 'pad' instead.", DeprecationWarning)
        pad = padding

    # Validate input dimensions
    input = as_tensor_variable(input)
    if input.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions.")

    # Default values for stride and pad
    if stride is None:
        stride = ws
    if pad is None:
        pad = (0, 0)

    # Convert inputs to tensors
    ws = as_tensor_variable(ws)
    stride = as_tensor_variable(stride)
    pad = as_tensor_variable(pad)

    # Validate parameter types
    if ws.dtype not in ("int32", "int64"):
        raise TypeError("Window size (`ws`) must be an integer type.")
    if stride.dtype not in ("int32", "int64"):
        raise TypeError("Stride (`stride`) must be an integer type.")
    if pad.dtype not in ("int32", "int64"):
        raise TypeError("Padding (`pad`) must be an integer type.")

    # Handle ignore_border default behavior
    if ignore_border is None:
        warnings.warn(
            "Default value of 'ignore_border' will change to True in future versions.",
            FutureWarning,
        )
        ignore_border = False

    # Pad the input tensor
    if pad[0] > 0 or pad[1] > 0:
        input = tensor_pad(input, [(0, 0)] * (input.ndim - 2) + [pad, pad], mode="constant")

    # Compute output shape
    input_shape = shape(input)
    out_height = (
        floor((input_shape[-2] - ws[0]) / stride[0]) + 1
        if ignore_border
        else floor((input_shape[-2] - 1) / stride[0]) + 1
    )
    out_width = (
        floor((input_shape[-1] - ws[1]) / stride[1]) + 1
        if ignore_border
        else floor((input_shape[-1] - 1) / stride[1]) + 1
    )

    # Create indices for pooling regions
    row_indices = arange(0, out_height) * stride[0]
    col_indices = arange(0, out_width) * stride[1]
    rows, cols = row_indices.dimshuffle([0, "x"]), col_indices.dimshuffle(["x", 0])
    rows = rows + arange(ws[0]).dimshuffle(["x", 0])
    cols = cols + arange(ws[1]).dimshuffle([0, "x"])

    # Extract pooling regions
    pooling_regions = input[..., rows, cols]

    # Apply pooling operation
    if mode == "max":
        output = max(pooling_regions, axis=(-2, -1))
    elif mode == "sum":
        output = sum(pooling_regions, axis=(-2, -1))
    elif mode == "average_inc_pad":
        output = sum(pooling_regions, axis=(-2, -1)) / (ws[0] * ws[1])
    elif mode == "average_exc_pad":
        output = sum(pooling_regions, axis=(-2, -1)) / ((ws[0] - 2 * pad[0]) * (ws[1] - 2 * pad[1]))
    else:
        raise ValueError(f"Unsupported pooling mode: {mode}")

    return output
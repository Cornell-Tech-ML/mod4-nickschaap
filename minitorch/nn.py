from typing import Tuple, Optional, Union, TYPE_CHECKING

from . import operators
from .autodiff import Context
from .fast_ops import FastOps

from .tensor_functions import Function, rand, zeros

if TYPE_CHECKING:
    from .tensors import Tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: "Tensor", kernel: Tuple[int, int]) -> Tuple["Tensor", int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # Compute the new dimensions after splitting by the kernel
    new_height = height // kh
    new_width = width // kw

    # Reshape:
    #   (batch, channel, height, width)
    # -> (batch, channel, new_height, kh, new_width, kw)
    out = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Permute to bring the kernel height and width together at the end:
    #   (batch, channel, new_height, new_width, kh, kw)
    # We permute to: (batch, channel, new_height, new_width, kh, kw)
    # and then merge kh and kw into a single dimension
    out = out.permute(0, 1, 2, 4, 3, 5).contiguous()
    out = out.view(batch, channel, new_height, new_width, kh * kw)

    return out, new_height, new_width


def avgpool2d(input: "Tensor", kernel: Tuple[int, int]) -> "Tensor":
    """2D Average Pooling

    Given an input of shape (batch, channel, height, width) and a kernel size (kh, kw),
    this function applies average pooling. That is, it divides the input into (kh x kw) blocks
    and computes the mean for each block, reducing the height and width.

    Args:
    ----
        input (Tensor): Input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): The (kernel_height, kernel_width) of the pooling.

    Returns:
    -------
        Tensor: The result of average pooling with shape (batch, channel, new_height, new_width)

    """
    # Use the tile function to reshape the input into blocks of size kh*kw
    # tiled will have shape: (batch, channel, new_height, new_width, kh*kw)
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the mean over the last dimension (the kernel dimension)
    # This reduces the shape to (batch, channel, new_height, new_width)
    return tiled.mean(tiled.dims - 1).view(
        input.shape[0], input.shape[1], new_height, new_width
    )


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: "Tensor", dim: int) -> "Tensor":
    """Compute the argmax as a 1-hot tensor."""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t: "Tensor", dim: Optional["Tensor"] = None) -> "Tensor":
        """Forward of max should be max reduction"""
        ctx.save_for_backward(t, int(dim.item()) if dim is not None else None)
        if dim is None:
            return max_reduce(t.contiguous().view(int(operators.prod(t.shape))), 0)
        return max_reduce(t, int(dim.item()))

    @staticmethod
    def backward(
        ctx: Context, grad_output: "Tensor"
    ) -> Union[Tuple["Tensor", ...], "Tensor"]:
        """Backward of max should be argmax"""
        a, dim = ctx.saved_values
        if dim is not None:
            return grad_output * argmax(a, dim), zeros(a.shape)
        return (
            argmax(a.contiguous().view(int(operators.prod(a.shape))), 0).view(*a.shape)
            * grad_output
        )


def softmax(input: "Tensor", dim: int) -> "Tensor":
    r"""Compute the softmax as a tensor.

    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
    ----
        input : input tensor
        dim : dimension to apply softmax

    Returns:
    -------
        softmax tensor

    """
    exp = input.exp()
    return exp / exp.sum(dim)


def logsoftmax(input: "Tensor", dim: int) -> "Tensor":
    r"""Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
    ----
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
    -------
         log of softmax tensor

    """
    return input - input.exp().sum(dim).log()


def maxpool2d(input: "Tensor", kernel: Tuple[int, int]) -> "Tensor":
    r"""Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor : pooled tensor

    """
    tiled, new_height, new_width = tile(input, kernel)

    # Compute the max over the last dimension (the kernel dimension)
    # This reduces the shape to (batch, channel, new_height, new_width)
    return tiled.max(tiled.dims - 1).view(
        input.shape[0], input.shape[1], new_height, new_width
    )


def dropout(input: "Tensor", rate: float, ignore: bool = False) -> "Tensor":
    """Dropout positions based on random noise.

    Args:
    ----
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
    -------
        tensor with random positions dropped out

    """
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)

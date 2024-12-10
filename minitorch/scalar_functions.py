from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @staticmethod
    def forward(ctx: Context, *inps: float) -> float:
        """The backward function for the scalar function."""
        ...

    @staticmethod
    def backward(ctx: Context, *inps: float) -> float:
        """The backward function for the scalar function."""
        ...

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to a set of values and return a new scalar."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Addition function."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivate of the addition function."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Log function."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivate of the log function."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiply function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Multiply function."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivate of the multiply function."""
        (a, b) = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Inverse function."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivate of the inverse function."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Exponential function."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivate of the exponential function."""
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class Neg(ScalarFunction):
    """Negate function $f(x) = -1 * x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Negate function."""
        ctx.save_for_backward(a)
        return operators.neg(a)

    @staticmethod
    def backward(_ctx: Context, d_output: float) -> float:
        """Derivate of the negate function."""
        return -1 * d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Sigmoid function."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivate of the sigmoid function."""
        (a,) = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """ReLU function."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivate of the ReLU function."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class LT(ScalarFunction):
    """Less than function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Less than function."""
        ctx.save_for_backward(a, b)
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, _d_output: float) -> Tuple[float, ...]:
        """Derivate of less than function."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Equality function."""
        ctx.save_for_backward(a, b)
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, _d_output: float) -> Tuple[float, ...]:
        """Derivate of equality function."""
        return 0.0, 0.0

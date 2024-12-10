from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
    wrap_tuple,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __hash__(self) -> int:
        return self.unique_id

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def __add__(self, b: ScalarLike) -> Scalar:
        """Addition."""
        return Add.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtraction."""
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        """Negation."""
        return Neg.apply(self)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Less than comparison."""
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Greater than comparison."""
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Equality comparison."""
        return EQ.apply(self, b)

    def relu(self) -> Scalar:
        """Rectified linear unit function."""
        return ReLU.apply(self)

    def sigmoid(self) -> Scalar:
        """Sigmoid function."""
        return Sigmoid.apply(self)

    def exp(self) -> Scalar:
        """Exponential function."""
        return Exp.apply(self)

    def log(self) -> Scalar:
        """Natural logarithm."""
        return Log.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: float) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            # self.__setattr__("derivative", 0.0)
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant (no derivative needed)"""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of this variable in the computation graph."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: float) -> Iterable[Tuple[Variable, float]]:
        """Apply the chain rule to compute the derivatives of the parent variables."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None
        assert h.inputs is not None
        # Implement the chain_rule function in Scalar for functions of arbitrary arguments.
        # This function should be able to backward process a function by passing
        # it in a context and \(d\) and then collecting the local derivatives.
        # It should then pair these with the right variables and return them.
        # This function is also where we filter out constants that were used on the forward pass,
        # but do not need derivatives.
        d_inputs = wrap_tuple(h.last_fn.backward(h.ctx, d_output))
        return [
            (inp, d_input)
            for inp, d_input in zip(h.inputs, d_inputs)
            if not inp.is_constant()
        ]

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Args:
    ----
        f : function from n-scalars to 1-scalar.
        *scalars : n input scalar values.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )

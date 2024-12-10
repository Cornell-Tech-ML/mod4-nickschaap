"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, TypeVar, List, Union, Sequence

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function. Return the input."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return float(-1 * x)


def lt(x: float, y: float) -> bool:
    """Check if x is less than y."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x is equal to y."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x is close to y."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Rectified Linear Unit function."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Exponential function."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Backward pass for natural logarithm."""
    return y / x


def inv(x: float) -> float:
    """Reciprocal function."""
    return 1.0 / x


def inv_back(x: float, b: float) -> float:
    """Backward pass for reciprocal."""
    return -1.0 * b / (x**2)


def relu_back(x: float, b: float) -> float:
    """Backward pass for ReLU."""
    return b if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def map(xs: Iterable[A], f: Callable[[A], B]) -> Iterable[B]:
    """Map a function `f` over a list `xs`."""
    return [f(x) for x in xs]


def zipWith(xs: Iterable[A], ys: Iterable[B], f: Callable[[A, B], C]) -> Iterable[C]:
    """Zip two lists `xs` and `ys` applying a function `f`."""
    return [f(x, y) for x, y in zip(xs, ys, strict=True)]


def reduce(xs: Iterable[A], f: Callable[[B, A], B], init: B) -> B:
    """Reduce a list `xs` using a function `f` and an initial value `init`."""
    result = init
    for x in xs:
        result = f(result, x)
    return result


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list of numbers."""
    return map(xs, neg)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists together."""
    return zipWith(xs, ys, lambda x, y: add(x, y))


def sum(xs: Iterable[float]) -> float:
    """Sum a list of numbers."""
    return reduce(xs, add, 0)


def prod(xs: Union[Sequence[int], Sequence[float]]) -> float:
    """Product of a list of numbers."""
    if len(xs) == 0:
        return 0
    return reduce(xs, mul, 1)


def matmult(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication."""
    return [
        [sum([a[i][k] * b[k][j] for k in range(len(a[0]))]) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def matadd(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Matrix addition."""
    return [[add(a[i][j], b[i][j]) for j in range(len(a[0]))] for i in range(len(a))]

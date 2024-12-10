from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Protocol, Callable, Dict, Set, Any

# ## Task 1.1
# Central Difference calculation


def central_difference(
    f: Callable[..., Any],
    *vals: Any,
    arg: int = 0,
    epsilon: float = 1e-6,
) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    if len(vals) <= arg:
        raise ValueError("arg is out of bounds")
    if type(vals) is not tuple:
        raise ValueError("vals is not a tuple")

    f_x_plus_epsilon = list(vals)
    f_x_minus_epsilon = list(vals)
    f_x_plus_epsilon[arg] = f_x_plus_epsilon[arg] + epsilon
    f_x_minus_epsilon[arg] = f_x_minus_epsilon[arg] - epsilon
    return (f(*f_x_plus_epsilon) - f(*f_x_minus_epsilon)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative with respect to this variable.

        Args:
        ----
        x (Any): The value to accumulate.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Get the unique identifier for this variable.

        Returns
        -------
        int: The unique identifier.

        """
        ...

    def is_leaf(self) -> bool:
        """Check if this variable is a leaf node in the computation graph.

        Returns
        -------
        bool: True if this variable is a leaf node, False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Check if this variable is a constant.

        Returns
        -------
        bool: True if this variable is a constant, False otherwise.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parent variables of this variable in the computation graph.

        Returns
        -------
        Iterable[Variable]: An iterable of parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Apply the chain rule to compute the derivatives of the parent variables.

        Args:
        ----
        d_output (Any): The derivative of the output with respect to this variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, each containing a parent variable and its corresponding derivative.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    order: Iterable[Variable] = []
    seen: Set[Variable] = set()

    def visit(var: Variable) -> None:
        if var in seen or var.is_constant():
            return

        if not var.is_leaf():
            for parent in var.parents:
                visit(parent)
        seen.add(var)
        order.insert(0, var)

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
    variable: The right-most variable.
    deriv: The derivative of the output with respect to the variable that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    queue: Iterable[Variable] = topological_sort(variable)
    derivatives: Dict[Variable, Any] = {}
    derivatives[variable] = deriv
    for var in queue:
        d_var = derivatives[var]
        if var.is_leaf():
            var.accumulate_derivative(d_var)
        else:
            for parent, parent_deriv in var.chain_rule(d_var):
                if parent not in derivatives:
                    derivatives[parent] = 0
                derivatives[parent] += parent_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Get the saved values."""
        return self.saved_values

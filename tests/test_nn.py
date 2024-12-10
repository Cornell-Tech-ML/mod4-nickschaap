import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


def all_close(a: Tensor, b: Tensor) -> bool:
    assert a.shape == b.shape, f"Shape mismatch {a.shape} != {b.shape}"
    return a.is_close(b).all().item() == 1.0


def test_tile() -> None:
    a = minitorch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    out, new_height, new_width = minitorch.tile(a, (2, 2))
    assert new_height == 1
    assert new_width == 1
    assert out.shape == (1, 1, 1, 1, 4)


def test_avg_pool() -> None:
    a = minitorch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    out = minitorch.avgpool2d(a, (2, 2))
    assert out.shape == (1, 1, 1, 1)

    assert_close(out[0, 0, 0, 0], 2.5)


def test_argmax() -> None:
    a = minitorch.tensor([0, 1, 2, 3, 4, 5])
    out = minitorch.argmax(a, 0)
    assert out == 5


def test_max_grad() -> None:
    a = minitorch.tensor([0, 1, 2, 3, 4, 5])
    b = a.max()
    b.backward()
    assert a.grad is not None
    assert a.grad == minitorch.tensor([0, 0, 0, 0, 0, 1])


def test_max_grad_2() -> None:
    a = minitorch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], requires_grad=True)
    b = a.max()
    b.backward()
    assert a.grad is not None

    assert all_close(minitorch.tensor([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]), a.grad)


def test_max_grad_3() -> None:
    a = minitorch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], requires_grad=True)
    b = a.max(0)
    b.backward(minitorch.zeros(b.shape) + 1)
    assert a.grad is not None

    assert all_close(minitorch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]), a.grad)


def test_max_grad_4() -> None:
    a = minitorch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], requires_grad=True)
    b = a.max(1)

    b.backward(minitorch.zeros(b.shape) + 1)
    assert a.grad is not None

    assert all_close(minitorch.tensor([[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]), a.grad)


def test_eq() -> None:
    a = minitorch.tensor([0, 1, 2, 3, 4, 5])
    b = minitorch.tensor([0, 1, 2, 3, 4, 5])
    assert all_close(a, b)

    c = minitorch.tensor([0, 1, 2, 3, 4, 6])
    assert not all_close(a, c)


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    t[0, 0, 0] = 1e9
    t[0, 1, 0] = 1e9
    t[0, 2, 0] = 1e9
    out = t.max()
    assert_close(out.item(), 1e9)
    assert out.shape == (1,)
    reduced = t.max(dim=0)

    assert reduced.shape == (1, 3, 4)
    assert_close(reduced[0, 0, 0], 1e9)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)

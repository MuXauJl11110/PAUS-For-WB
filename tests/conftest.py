import pytest
import torch

from src.oracles.oracle import OperatorOracle
from src.oracles.point import Point

d_list: list[int] = [250, 500]
n_list: list[int] = [50, 100]
T_list: list[int] = [200, 300]
full_list: list[bool] = [True, False]


@pytest.fixture(params=d_list)
def d(request):
    return request.param


@pytest.fixture(params=n_list)
def n(request):
    return request.param


@pytest.fixture(params=T_list)
def T(request):
    return request.param


@pytest.fixture(params=full_list)
def full(request):
    return request.param


@pytest.fixture
def C(d: int):
    return torch.randn(d, d)


@pytest.fixture
def x(d: int, T: int):
    return torch.randn((T, d, d))


@pytest.fixture
def u(d: int, T: int):
    output = torch.randn((T, d))
    return output / output.max()


@pytest.fixture
def v(d: int, T: int):
    output = torch.randn((T, d))
    return output / output.max()


@pytest.fixture
def p(d: int):
    output = torch.rand(d)
    return output / output.sum()


@pytest.fixture
def q(d: int, T: int):
    output = torch.rand((T, d))
    return output / output.sum(dim=1)[:, None]


@pytest.fixture
def z(x: torch.Tensor, p: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    return Point(x, p, u, v)


@pytest.fixture
def oracle(C: torch.Tensor, q: torch.Tensor, n: int, full: bool):
    return OperatorOracle(C, q, n, full)


@pytest.fixture
def A(d: int):
    return torch.cat(
        (
            torch.kron(torch.eye(d), torch.ones((1, d))),
            torch.kron(torch.ones((1, d)), torch.eye(d)),
        )
    ).to_sparse()

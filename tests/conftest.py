import jax
import jax.numpy as jnp
import numpy as onp
import pytest
import torch

from src.oracles.oracle import OperatorOracle
from src.oracles.point import Point

d_list: list[int] = [250, 500]
n_list: list[int] = [50, 100]
T_list: list[int] = [200, 300]
gamma_list: list[float] = [0.001]
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


@pytest.fixture(params=gamma_list)
def gamma(request):
    return request.param


@pytest.fixture(params=full_list)
def full(request):
    return request.param


@pytest.fixture
def C(d: int):
    return torch.randn(d, d)


@pytest.fixture
def x(d: int, T: int):
    output = torch.rand((T, d, d))
    output /= output.sum(dim=(1, 2))[:, None, None]
    return output


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
    return Point(torch.log(x), torch.log(p), u, v)


@pytest.fixture
def oracle(C: torch.Tensor, q: torch.Tensor, n: int, full: bool, gamma: float):
    return OperatorOracle(C, q, n, full, gamma)


@pytest.fixture
def A(d: int):
    return torch.cat(
        (
            torch.kron(torch.eye(d), torch.ones((1, d))),
            torch.kron(torch.ones((1, d)), torch.eye(d)),
        )
    ).to_sparse()


@pytest.fixture
def jax_f_and_grad(
    oracle: OperatorOracle,
    gamma: float,
    z: Point,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, p, u, v, q, denominator = oracle._preprocess_z(z)
    x_jax = jnp.array(x)
    u_jax = jnp.array(u)
    v_jax = jnp.array(v)
    p_jax = jnp.array(p)
    q_jax = jnp.array(q)
    C_jax = jnp.array(oracle._C)

    def f(
        x: jnp.ndarray,
        p: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        q: jnp.ndarray,
        denominator: jnp.ndarray,
        C: jnp.ndarray,
    ):
        C_norm = jnp.max(C)
        result = jnp.sum(C[None, :, :] * x) + 2 * C_norm * jnp.sum(
            u * (x.sum(axis=2) - p[None, :]) + v * (x.sum(axis=1) - q)
        )
        result /= denominator
        result += 0.5 * gamma * (jnp.sum(x * x) + jnp.sum(p * p))
        return result

    f_num = f(x_jax, p_jax, u_jax, v_jax, q_jax, denominator, C_jax)

    grad_f = jax.grad(f, argnums=(0, 1, 2, 3))
    grad_x, grad_p, grad_u, grad_v = grad_f(x_jax, p_jax, u_jax, v_jax, q_jax, denominator, C_jax)
    return (
        torch.from_numpy(onp.array(f_num)),
        torch.from_numpy(onp.array(grad_x)),
        torch.from_numpy(onp.array(grad_p)),
        torch.from_numpy(onp.array(grad_u)),
        torch.from_numpy(onp.array(grad_v)),
    )

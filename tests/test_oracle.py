import torch

from src.oracles.oracle import OperatorOracle
from src.oracles.point import Point


def test_oracle_f(oracle: OperatorOracle, z: Point, A: torch.Tensor):
    f = oracle.f(z)
    d, T, n = oracle._d, oracle._T, oracle._n
    q, C, C_norm = oracle._q, oracle._C, oracle._C_norm

    if oracle._full:
        x, p, u, v = z.x, z.p, z.v, z.u
        denominator = T
    else:
        x, p, u, v = z.x[:n], z.p, z.v[:n], z.u[:n]
        denominator = n

    _f = 0.0
    for u_i, v_i, x_i, q_i in zip(u, v, x, q):
        b_i = torch.cat((p, q_i))
        y_i = torch.cat((u_i, v_i))
        X_i = x_i.view(d * d)
        _f += torch.sum(C * x_i) + 2 * C_norm * torch.sum(torch.dot(y_i, A @ X_i) - torch.dot(b_i, y_i))

    _f /= denominator
    assert torch.allclose(f, _f, rtol=1e-4)


def test_oracle_grad_x(oracle: OperatorOracle, z: Point, A: torch.Tensor):
    grad = oracle.grad_x(z)
    d, T, n = oracle._d, oracle._T, oracle._n
    q, C, C_norm = oracle._q, oracle._C, oracle._C_norm

    if oracle._full:
        x, p, u, v = z.x, z.p, z.v, z.u
        denominator = T
    else:
        x, p, u, v = z.x[:n], z.p, z.v[:n], z.u[:n]
        denominator = n

    _grad = torch.zeros((T, d, d))
    for i, (u_i, v_i) in enumerate(zip(u, v)):
        y_i = torch.cat((u_i, v_i))
        _grad[i, :, :] = (C + 2 * C_norm * (A.T @ y_i).view(d, d)) / denominator

    assert torch.allclose(grad, _grad, rtol=1e-4)


def test_oracle_grad_p(oracle: OperatorOracle, z: Point, A: torch.Tensor):
    grad = oracle.grad_p(z)
    d, T, n = oracle._d, oracle._T, oracle._n

    if oracle._full:
        _, _, u, _ = z.x, z.p, z.v, z.u
        denominator = T
    else:
        _, _, u, _ = z.x[:n], z.p, z.v[:n], z.u[:n]
        denominator = n

    _grad = torch.zeros(d)
    for u_i in u:
        _grad += -u_i

    _grad /= denominator
    assert torch.allclose(grad, _grad, rtol=1e-4)


def test_oracle_grad_uv(oracle: OperatorOracle, z: Point, A: torch.Tensor):
    grad_u, grad_v = oracle.grad_u(z), oracle.grad_v(z)
    d, T, n = oracle._d, oracle._T, oracle._n
    q, _, C_norm = oracle._q, oracle._C, oracle._C_norm

    if oracle._full:
        x, p, u, v = z.x, z.p, z.v, z.u
        denominator = T
    else:
        x, p, u, v = z.x[:n], z.p, z.v[:n], z.u[:n]
        denominator = n

    _grad_u, _grad_v = torch.zeros((T, d)), torch.zeros((T, d))
    for i, (_, _, x_i, q_i) in enumerate(zip(u, v, x, q)):
        b_i = torch.cat((p, q_i))
        grad_y = 2 * C_norm * (A @ x_i.view(d * d) - b_i) / denominator
        _grad_u[i, :] = grad_y[:d]
        _grad_v[i, :] = grad_y[d:]

    assert torch.allclose(grad_u, _grad_u, rtol=1e-1)  # accumulation error
    assert torch.allclose(grad_v, _grad_v, rtol=1e-1)  # accumulation error

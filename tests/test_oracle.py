import torch

from src.oracles.oracle import OperatorOracle
from src.oracles.point import Point


def test_oracle_f(oracle: OperatorOracle, z: Point, full: bool):
    f = oracle.f(z, full)
    d, T, n = oracle._d, oracle._T, oracle._n
    q, C, C_norm = oracle._q, oracle._C, oracle._C_norm

    A = torch.cat(
        (
            torch.kron(torch.eye(d), torch.ones((1, d))),
            torch.kron(torch.ones((1, d)), torch.eye(d)),
        )
    ).to_sparse()
    if full:
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

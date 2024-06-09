import torch

from src.oracles.point import BaseOracle, Point


class OperatorOracle(BaseOracle):
    def __init__(self, C: torch.Tensor, q: torch.Tensor, n: int, full: bool = True):
        """
        :param torch.Tensor C: Cost matrix of shape (d, d).
        :param torch.Tensor q: Marginal distributions of shape (T, d).
        :param int n: Number of samples on server's node.
        :param bool full: If True then full operator computed, otherwise only for server part, defaults to True
        """
        self._T, self._d = q.shape
        self._n = n
        self._C = C
        self._C_norm = torch.max(self._C)
        self._q = q
        self._full = full
        assert self._n <= self._T
        assert C.shape == (self._d, self._d)

    def f(self, z: Point) -> float:
        x, p, u, v, q, denominator = self._preprocess_z(z)

        result = torch.sum(self._C[None, :, :] * x) + 2 * self._C_norm * torch.sum(
            u * (x.sum(dim=2) - p[None, :]) + v * (x.sum(dim=1) - q)
        )

        return result / denominator

    def grad_x(self, z: Point) -> torch.Tensor:
        _, _, u, v, _, denominator = self._preprocess_z(z)
        grad = self._C.repeat(denominator, 1, 1) + 2 * self._C_norm * (u[:, :, None] + v[:, None, :])

        return torch.cat((grad, torch.zeros((self._T - denominator, self._d, self._d)))) / denominator

    def grad_p(self, z: Point) -> torch.Tensor:
        _, _, u, _, _, denominator = self._preprocess_z(z)

        return -u.sum(dim=0) / denominator

    def grad_u(self, z: Point) -> torch.Tensor:
        x, p, _, _, _, denominator = self._preprocess_z(z)
        grad = 2 * self._C_norm * (x.sum(dim=2) - p[None, :])

        return torch.cat((grad, torch.zeros((self._T - denominator, self._d)))) / denominator

    def grad_v(self, z: Point) -> torch.Tensor:
        x, _, _, _, q, denominator = self._preprocess_z(z)
        grad = 2 * self._C_norm * (x.sum(dim=1) - q)

        return torch.cat((grad, torch.zeros((self._T - denominator, self._d)))) / denominator

    def _preprocess_z(self, z: Point) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._full:
            x, p, u, v = z.x, z.p, z.v, z.u
            q = self._q
            denominator = self._T
        else:
            x, p, u, v = z.x[: self._n], z.p, z.v[: self._n], z.u[: self._n]
            q = self._q[: self._n]
            denominator = self._n

        return x, p, u, v, q, denominator

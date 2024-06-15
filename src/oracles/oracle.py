from abc import ABC, abstractmethod

import torch

from src.oracles.point import OperatorPoint, Point


class BaseOracle(ABC):
    r"""
    Base class for implementation operator oracles.
    """

    @abstractmethod
    def f(self, z: Point) -> float:
        """
        Computes the value of operator at point z.
        """
        pass

    @abstractmethod
    def grad_x(self, z: Point) -> torch.Tensor:
        """
        Computes the value of x gradient at point z.
        """
        pass

    @abstractmethod
    def grad_p(self, z: Point) -> torch.Tensor:
        """
        Computes the value of p gradient at point z.
        """
        pass

    @abstractmethod
    def grad_u(self, z: Point) -> torch.Tensor:
        """
        Computes the value of u gradient at point z.
        """
        pass

    @abstractmethod
    def grad_v(self, z: Point) -> torch.Tensor:
        """
        Computes the value of v gradient at point z.
        """
        pass

    def G(self, z: Point) -> OperatorPoint:
        """
        Computes the value of operator (grad_x(z), -grad_y(z)) at point z.
        """
        return OperatorPoint(self.grad_x(z), self.grad_p(z), -self.grad_u(z), -self.grad_v(z))


class OperatorOracle(BaseOracle):
    def __init__(self, C: torch.Tensor, q: torch.Tensor, n: int, full: bool = True, gamma: float = 0.001):
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
        self._gamma = gamma
        assert self._n <= self._T
        assert C.shape == (self._d, self._d)

    def f(self, z: Point) -> float:
        x, p, u, v, q, denominator = self._preprocess_z(z)

        result = torch.sum(self._C[None, :, :] * x) + 2 * self._C_norm * torch.sum(
            u * (x.sum(dim=2) - p[None, :]) + v * (x.sum(dim=1) - q)
        )
        result /= denominator
        result += 0.5 * self._gamma * (torch.sum(p * p) + torch.sum(x * x))

        return result

    def grad_x(self, z: Point) -> torch.Tensor:
        x, _, u, v, _, denominator = self._preprocess_z(z)
        grad = self._C.repeat(denominator, 1, 1) + 2 * self._C_norm * (u[:, :, None] + v[:, None, :])
        grad /= denominator
        grad += self._gamma * x

        return torch.cat((grad, torch.zeros((self._T - denominator, self._d, self._d))))

    def grad_p(self, z: Point) -> torch.Tensor:
        _, p, u, _, _, denominator = self._preprocess_z(z)

        return -2 * self._C_norm * u.sum(dim=0) / denominator + self._gamma * p
        # return -2 * self._C_norm * u.sum(dim=0) / denominator

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
            x, p, u, v = torch.exp(z.log_x), torch.exp(z.log_p), z.u, z.v
            q = self._q
            denominator = self._T
        else:
            x, p, u, v = torch.exp(z.log_x[: self._n]), torch.exp(z.log_p), z.u[: self._n], z.v[: self._n]
            q = self._q[: self._n]
            denominator = self._n

        return x, p, u, v, q, denominator

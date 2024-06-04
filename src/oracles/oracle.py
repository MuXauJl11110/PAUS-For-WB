import torch

from src.oracles.point import BaseOracle, Point


class OperatorOracle(BaseOracle):
    def __init__(self, C: torch.Tensor, q: torch.Tensor, n: int):
        """
        :param torch.Tensor C: Cost matrix of shape (d, d).
        :param torch.Tensor q: Marginal distributions of shape (T, d).
        :param int n: _description_
        """
        self._T, self._d = q.shape
        self._n = n
        self._C = C
        self._C_norm = torch.max(self._C)
        self._q = q
        assert self._n <= self._T
        assert C.shape == (self._d, self._d)

        self._grad_x: torch.Tensor = torch.zeros((self._T, self._d, self._d))
        self._grad_p: torch.Tensor = torch.zeros(self._d)
        self._grad_u: torch.Tensor = torch.zeros((self._T, self._d))
        self._grad_v: torch.Tensor = torch.zeros((self._T, self._d))
        self._grad: Point = Point(self._grad_x, self._grad_p, self._grad_u, self._grad_v)

    def f(self, z: Point, full: bool = True) -> float:
        """
        :param Point z: Point at which function is computed
        :param bool full: If True then full operator computed, otherwise only for server part, defaults to True
        """
        if full:
            x, p, u, v = z.x, z.p, z.v, z.u
            q = self._q
            denominator = self._T
        else:
            x, p, u, v = z.x[: self._n], z.p, z.v[: self._n], z.u[: self._n]
            q = self._q[: self._n]
            denominator = self._n
        result = torch.sum(self._C[None, :, :] * x) + 2 * self._C_norm * torch.sum(
            u * (x.sum(dim=2) - p[None, :]) + v * (x.sum(dim=1) - q)
        )

        return result / denominator

    def grad_x(self, z: Point) -> torch.Tensor:
        pass
        # self._grad_x = self._C
        # for i in range(self._n):  # tnrange(self._n, desc="Grad x"):
        #     self._grad_x[i] = self._oracles[i].grad_x(Point(z.x[i], z.p, z.u[i], z.v[i])) / self._n

        # return self._grad_x

    def grad_p(self, z: Point) -> torch.Tensor:
        pass
        # grad_p = np.zeros_like(self._grad_p)
        # for i in range(self._n):  #  tnrange(self._n, desc="Grad p"):
        #     grad_p += self._oracles[i].grad_p(Point(z.x[i], z.p, z.u[i], z.v[i]))

        # self._grad_p = grad_p / self._n
        # return self._grad_p

    def grad_u(self, z: Point) -> torch.Tensor:
        pass
        # for i in range(self._n):  # tnrange(self._n, desc="Grad u"):
        #     self._grad_u[i] = self._oracles[i].grad_u(Point(z.x[i], z.p, z.u[i], z.v[i])) / self._n

        # return self._grad_u

    def grad_v(self, z: Point) -> torch.Tensor:
        pass
        # for i in range(self._n):  # tnrange(self._n, desc="Grad v"):
        #     self._grad_v[i] = self._oracles[i].grad_v(Point(z.x[i], z.p, z.u[i], z.v[i])) / self._n

        # return self._grad_v

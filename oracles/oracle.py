import numpy as np
from tqdm.notebook import tnrange

from oracles.point import HPointType, OTProblemOracle, Point
from oracles.space import BaseSpaceOracle, HSpaceType, SpacePoint, XSpaceType


class OperatorOracle(BaseSpaceOracle):
    def __init__(self, oracles: list[OTProblemOracle], d: int, n: int):
        """
        :param list[OTProblemOracle] oracles: Oracles.
        :param int d: Histogram size.
        :param int n: Number of instances to compute.
        """
        self._T = len(oracles)
        self._n = n
        assert self._n <= self._T
        self._oracles = oracles
        self._grad_x: XSpaceType = np.zeros((self._T, d, d))
        self._grad_p: HPointType = np.zeros(d)
        self._grad_u: HSpaceType = np.zeros((self._T, d))
        self._grad_v: HSpaceType = np.zeros((self._T, d))
        self._grad: SpacePoint = SpacePoint(self._grad_x, self._grad_p, self._grad_u, self._grad_v)

    def f(self, z: SpacePoint) -> float:
        output = 0.0
        for i in tnrange(self._n):
            output += self._oracles[i].f(Point(z.x[i], z.p, z.u[i], z.v[i]))

        return output / self._n

    def grad_x(self, z: SpacePoint) -> XSpaceType:
        for i in tnrange(self._n):  # , desc="Grad x"):
            self._grad_x[i] = self._oracles[i].grad_x(Point(z.x[i], z.p, z.u[i], z.v[i]))

        self._grad_x /= self._n
        return self._grad_x

    def grad_p(self, z: SpacePoint) -> HPointType:
        grad_p = np.zeros_like(self._grad_p)
        for i in tnrange(self._n):  # , desc="Grad p"):
            grad_p += self._oracles[i].grad_p(Point(z.x[i], z.p, z.u[i], z.v[i]))

        self._grad_p /= self._n
        return self._grad_p

    def grad_u(self, z: SpacePoint) -> HSpaceType:
        for i in tnrange(self._n):  # , desc="Grad u"):
            self._grad_u[i] = self._oracles[i].grad_u(Point(z.x[i], z.p, z.u[i], z.v[i]))

        self._grad_u /= self._n
        return self._grad_u

    def grad_v(self, z: SpacePoint) -> HSpaceType:
        for i in tnrange(self._n):  # , desc="Grad v"):
            self._grad_v[i] = self._oracles[i].grad_v(Point(z.x[i], z.p, z.u[i], z.v[i]))

        self._grad_v /= self._n
        return self._grad_v

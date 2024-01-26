from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.float64)
HPointType = Annotated[npt.NDArray[DType], Literal["d"]]  # histogram dimension of d
XPointType = Annotated[npt.NDArray[DType], Literal["d", "d"]]  # x dimension of d * d


class Point:
    def __init__(self, x: XPointType, p: HPointType, u: HPointType, v: HPointType) -> None:
        """
        :param XPointType x: Primal variable.
        :param HPointType p: Primal variable.
        :param HPointType u: Dual variable.
        :param HPointType v: Dual variable.
        """
        self.x: XPointType = x
        self.p: HPointType = p
        self.u: HPointType = u
        self.v: HPointType = v

    def __iadd__(self, z: Point) -> Point:
        self.x += z.x
        self.p += z.p
        self.u += z.v
        self.v += z.v
        return self

    def __add__(self, z: Point) -> Point:
        new_z = Point(self.x, self.p, self.u, self.v)
        new_z += z
        return new_z

    def __isub__(self, z: Point) -> Point:
        self.x -= z.x
        self.p -= z.p
        self.u -= z.u
        self.v -= z.v
        return self

    def __sub__(self, z: Point) -> Point:
        new_z = Point(self.x, self.p, self.u, self.v)
        new_z -= z
        return new_z


class BasePointOracle(ABC):
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
    def grad_x(self, z: Point) -> XPointType:
        """
        Computes the value of x gradient at point z.
        """
        pass

    @abstractmethod
    def grad_p(self, z: Point) -> HPointType:
        """
        Computes the value of p gradient at point z.
        """
        pass

    @abstractmethod
    def grad_u(self, z: Point) -> HPointType:
        """
        Computes the value of u gradient at point z.
        """
        pass

    @abstractmethod
    def grad_v(self, z: Point) -> HPointType:
        """
        Computes the value of v gradient at point z.
        """
        pass

    def G(self, z: Point) -> Point:
        """
        Computes the value of operator (grad_x(z), -grad_y(z)) at point z.
        """
        return Point(self.grad_x(z), self.grad_p(z), -self.grad_u(z), -self.grad_v(z))


class OTProblemOracle(BasePointOracle):
    def __init__(self, q: HPointType, C: XPointType) -> None:
        self.q = q
        self.C = C
        self.C_norm = np.max(np.abs(self.C))
        self.one = np.ones_like(q)

    def f(self, z: Point) -> float:
        """Lagrangian function at point (p, z)."""
        return (self.C * z.x).sum() + np.dot(z.u, z.x.sum(axis=0) - z.p) + np.dot(z.v, z.x.sum(axis=1) - self.q)

    def grad_x(self, z: Point) -> XPointType:
        return self.C + z.x + np.outer(z.u, self.one) + np.outer(self.one, z.v)

    def grad_p(self, z: Point) -> HPointType:
        return -z.u

    def grad_u(self, z: Point) -> HPointType:
        return z.x.sum(axis=0) - z.p

    def grad_v(self, z: Point) -> HPointType:
        return z.x.sum(axis=1) - self.q

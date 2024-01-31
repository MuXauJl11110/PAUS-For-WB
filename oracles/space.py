from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal

import numpy.typing as npt

from oracles.point import DType, HPointType

XSpaceType = Annotated[npt.NDArray[DType], Literal["T", "d", "d"]]  # x dimension of T * d * 2
HSpaceType = Annotated[npt.NDArray[DType], Literal["T", "d"]]  # y dimension T * d


class SpacePoint:
    def __init__(self, x: XSpaceType, p: HPointType, u: HSpaceType, v: HSpaceType) -> None:
        self.x: XSpaceType = x
        self.p: HPointType = p
        self.u: HSpaceType = u
        self.v: HSpaceType = v

    def __iadd__(self, z: SpacePoint) -> SpacePoint:
        self.x += z.x
        self.p += z.p
        self.u += z.u
        self.v += z.v
        return self

    def __add__(self, z: SpacePoint) -> SpacePoint:
        return SpacePoint(self.x + z.x, self.p + z.p, self.u + z.u, self.v + z.v)

    def __isub__(self, z: SpacePoint) -> SpacePoint:
        self.x -= z.x
        self.p -= z.p
        self.u -= z.u
        self.v -= z.v
        return self

    def __sub__(self, z: SpacePoint) -> SpacePoint:
        return SpacePoint(self.x - z.x, self.p - z.p, self.u - z.u, self.v - z.v)


class BaseSpaceOracle(ABC):
    r"""
    Base class for implementation operator oracles.
    """

    @abstractmethod
    def f(self, z: SpacePoint) -> float:
        """
        Computes the value of operator at point z.
        """
        pass

    @abstractmethod
    def grad_x(self, z: SpacePoint) -> XSpaceType:
        """
        Computes the value of x gradient at point z.
        """
        pass

    @abstractmethod
    def grad_p(self, z: SpacePoint) -> HPointType:
        """
        Computes the value of p gradient at point z.
        """
        pass

    @abstractmethod
    def grad_u(self, z: SpacePoint) -> HPointType:
        """
        Computes the value of y gradient at point z.
        """
        pass

    @abstractmethod
    def grad_v(self, z: SpacePoint) -> HPointType:
        """
        Computes the value of y gradient at point z.
        """
        pass

    def G(self, z: SpacePoint) -> SpacePoint:
        """
        Computes the value of operator (grad_x(z), -grad_y(z)) at point z.
        """
        return SpacePoint(self.grad_x(z), self.grad_p(z), -self.grad_u(z), -self.grad_v(z))

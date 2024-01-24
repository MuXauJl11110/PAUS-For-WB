from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt

from oracles.point import DType, HPointType, XPointType, YPointType

XSpaceType = Annotated[npt.NDArray[DType], Literal["T", "d^2"]]  # x dimension of T * d^2
YSpaceType = Annotated[npt.NDArray[DType], Literal["T", "2d"]]  # y dimension T * 2d


class SpacePoint:
    def __init__(self, xs: list[XPointType], p: HPointType, ys: list[YPointType]) -> None:
        self._x: XSpaceType = np.array(xs)
        self._p: HPointType = p
        self._y: YSpaceType = np.array(ys)

    @property
    def x(self) -> XSpaceType:
        return self._x

    @x.setter
    def x(self, new_x: XSpaceType) -> None:
        self._x = new_x

    @property
    def y(self) -> YSpaceType:
        return self._y

    @y.setter
    def y(self, new_y: YSpaceType) -> None:
        self._y = new_y

    @property
    def p(self) -> HPointType:
        return self._p

    @p.setter
    def p(self, new_p: HPointType) -> None:
        self._p = new_p

    def __iadd__(self, z: SpacePoint) -> SpacePoint:
        self._x += z.x
        self._p += z.p
        self._y += z.y
        return self

    def __add__(self, z: SpacePoint) -> SpacePoint:
        new_z = SpacePoint(self._x.tolist(), self._p, self._y.tolist())
        new_z += z
        return new_z

    def __isub__(self, z: SpacePoint) -> SpacePoint:
        self._x -= z.x
        self._p -= z.p
        self._y -= z.y
        return self

    def __sub__(self, z: SpacePoint) -> SpacePoint:
        new_z = SpacePoint(self._x.tolist(), self._p, self._y.tolist())
        new_z -= z
        return new_z


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
    def grad_y(self, z: SpacePoint) -> YPointType:
        """
        Computes the value of y gradient at point z.
        """
        pass

    def G(self, z: SpacePoint) -> SpacePoint:
        """
        Computes the value of operator (grad_x(z), -grad_y(z)) at point z.
        """
        return SpacePoint(self.grad_x(z).tolist(), self.grad_p(z), -self.grad_y(z).tolist())

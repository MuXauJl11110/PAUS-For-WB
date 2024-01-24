from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.float64)
HPointType = Annotated[npt.NDArray[DType], Literal["d"]]  # histogram dimension of d
XPointType = Annotated[npt.NDArray[DType], Literal["d^2"]]  # x dimension of d^2
YPointType = Annotated[npt.NDArray[DType], Literal["2d"]]  # y dimension 2d


class Point:
    def __init__(self, x: XPointType, y: YPointType) -> None:
        self._x: XPointType = x
        self._y: YPointType = y

    @property
    def x(self) -> XPointType:
        return self._x

    @x.setter
    def x(self, new_x: XPointType) -> None:
        self._x = new_x

    @property
    def y(self) -> YPointType:
        return self._y

    @y.setter
    def y(self, new_y: YPointType) -> None:
        self._y = new_y

    def __iadd__(self, z: Point) -> Point:
        self._x += z.x
        self._y += z.y
        return self

    def __add__(self, z: Point) -> Point:
        new_z = Point(self._x, self._y)
        new_z += z
        return new_z

    def __isub__(self, z: Point) -> Point:
        self._x -= z.x
        self._y -= z.y
        return self

    def __sub__(self, z: Point) -> Point:
        new_z = Point(self._x, self._y)
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
    def grad_y(self, z: Point) -> YPointType:
        """
        Computes the value of y gradient at point z.
        """
        pass

    @abstractmethod
    def G(self, z: Point) -> Point:
        """
        Computes the value of operator (grad_x(z), -grad_y(z)) at point z.
        """
        pass


AType = Annotated[npt.NDArray[DType], Literal["2d", "d^2"]]  # histogram dimension of (2d, d^2)


class WassersteinDistanceOracle(BasePointOracle):
    def __init__(self, p: HPointType, q: HPointType, d: XPointType, A: AType) -> None:
        self._b: XPointType = np.vstack((p, q))
        self._d: XPointType = d
        self._d_norm = np.max(np.abs(self._d))
        self._A: AType = A

    def f(self, z: Point) -> float:
        return np.dot(self._d, z.x) + 2 * self._d_norm * (z.y.T @ self._A @ z.x - np.dot(self._b, z.y))

    def grad_x(self, z: Point) -> XPointType:
        return self._d + 2 * self._d_norm * (self._A.T @ z.y)

    def grad_y(self, z: Point) -> YPointType:
        return 2 * self._d_norm * (self._A @ z.x - self._b)

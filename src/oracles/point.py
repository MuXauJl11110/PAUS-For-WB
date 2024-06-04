from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Point:
    def __init__(self, x: torch.Tensor, p: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> None:
        """
        :param torch.Tensor x: Primal variable of shape (T, d, d).
        :param torch.Tensor p: Primal variable of shape (1, d).
        :param torch.Tensor u: Dual variable of shape (T, d). # TODO: change to case when x has the shape (T, d1, d2)
        :param torch.Tensor v: Dual variable of shape (T, d).
        """
        self.x: torch.Tensor = x
        self.p: torch.Tensor = p
        self.u: torch.Tensor = u
        self.v: torch.Tensor = v

    def __iadd__(self, z: Point) -> Point:
        self.x += z.x
        self.p += z.p
        self.u += z.v
        self.v += z.v
        return self

    def __add__(self, z: Point) -> Point:
        return Point(self.x + z.x, self.p + z.p, self.u + z.u, self.v + z.v)

    def __isub__(self, z: Point) -> Point:
        self.x -= z.x
        self.p -= z.p
        self.u -= z.u
        self.v -= z.v
        return self

    def __sub__(self, z: Point) -> Point:
        return Point(self.x - z.x, self.p - z.p, self.u - z.u, self.v - z.v)


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

    def G(self, z: Point) -> Point:
        """
        Computes the value of operator (grad_x(z), -grad_y(z)) at point z.
        """
        return Point(self.grad_x(z), self.grad_p(z), -self.grad_u(z), -self.grad_v(z))

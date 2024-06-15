from __future__ import annotations

import torch


class Point:
    def __init__(
        self,
        log_x: torch.Tensor,
        log_p: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        :param torch.Tensor log_x: Logarithm of primal variable of shape (T, d, d).
        :param torch.Tensor log_p: Lograithm of primal variable of shape (1, d).
        :param torch.Tensor u: Dual variable of shape (T, d). # TODO: change to case when x has the shape (T, d1, d2)
        :param torch.Tensor v: Dual variable of shape (T, d).
        """
        self.log_x: torch.Tensor = log_x
        self.log_p: torch.Tensor = log_p
        self.u: torch.Tensor = u
        self.v: torch.Tensor = v

    def __iadd__(self, z: Point) -> Point:
        self.log_x += z.log_x
        self.log_p += z.log_p
        self.u += z.v
        self.v += z.v
        return self

    def __add__(self, z: Point) -> Point:
        return Point(self.log_x + z.log_x, self.log_p + z.log_p, self.u + z.u, self.v + z.v)

    def __isub__(self, z: Point) -> Point:
        self.log_x -= z.log_x
        self.log_p -= z.log_p
        self.u -= z.u
        self.v -= z.v
        return self

    def __sub__(self, z: Point) -> Point:
        return Point(self.log_x - z.log_x, self.log_p - z.log_p, self.u - z.u, self.v - z.v)


class OperatorPoint:
    def __init__(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
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

    def __iadd__(self, z: OperatorPoint) -> OperatorPoint:
        self.x += z.x
        self.p += z.p
        self.u += z.v
        self.v += z.v
        return self

    def __add__(self, z: OperatorPoint) -> OperatorPoint:
        return OperatorPoint(self.x + z.x, self.p + z.p, self.u + z.u, self.v + z.v)

    def __isub__(self, z: OperatorPoint) -> OperatorPoint:
        self.x -= z.x
        self.p -= z.p
        self.u -= z.u
        self.v -= z.v
        return self

    def __sub__(self, z: OperatorPoint) -> OperatorPoint:
        return OperatorPoint(self.x - z.x, self.p - z.p, self.u - z.u, self.v - z.v)

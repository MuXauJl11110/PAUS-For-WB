from collections import defaultdict

import numpy as np
import ot

from oracles import OperatorOracle, SpacePoint
from utils.space import init_space_point, project_onto_space


class PAUS:
    def __init__(
        self,
        F: OperatorOracle,
        F1: OperatorOracle,
        L_F1: float,
        composite_max_iters: int,
        log: bool = True,
        bar_true: np.ndarray | None = None,
    ):
        """
        :param OperatorOracle F: Operator of the problem.
        :param OperatorOracle F1: Operator of the server's part of the problem.
        :param float L_server: Parameter of Lipschitz-continuity of the server.
        :param int composite_max_iters: Number of composite steps.
        :param bool log: Logging, defaults to False
        """
        self.F = F
        self.F1 = F1
        self.L_F1 = L_F1
        self.composite_max_iters = composite_max_iters
        self.log = log

        self.bar_true = bar_true
        if self.bar_true is not None:
            self.dist_true = 0.0
            for oracle in self.F._oracles:
                self.dist_true += ot.emd2(bar_true, oracle.q, oracle.C)  # type: ignore
            self.dist_true /= len(self.F._oracles)

    def fit(
        self, delta: float, gamma: float | None = None, max_iter: int = 1000
    ) -> tuple[SpacePoint, dict[str, list[float]]]:
        history: dict[str, list[float]] = defaultdict(list)

        z_k = init_space_point(len(self.F._grad_p), len(self.F._grad_x))
        u_k = init_space_point(len(self.F._grad_p), len(self.F._grad_x))
        if gamma is None:
            gamma = 1 / delta

        for iter_num in range(max_iter):
            if self.log and iter_num % 10 == 0:
                print(f"Iter: {iter_num}")
            G_z_k = self.F.G(z_k) - self.F1.G(z_k)
            u_k = self.composite_mp(gamma, z_k, G_z_k)

            G_u_k = self.F.G(u_k) - self.F1.G(u_k)
            G = G_u_k + G_z_k
            z_k.x = u_k.x * np.exp(-gamma * G.x)
            z_k.p = u_k.p * np.exp(-gamma * G.p)
            z_k.u = u_k.u - gamma * G.u
            z_k.v = u_k.v - gamma * G.v
            z_k = project_onto_space(z_k)
            if self.log and self.bar_true is not None and (iter_num % 10 == 0):
                err = self.dual_gap(z_k.p)
                print(f"Err: {err}")
                history["err"].append(err)  # type: ignore

        return z_k, history

    def dual_gap(self, p: np.ndarray) -> float:
        dist = 0.0
        for oracle in self.F._oracles:
            dist += ot.emd2(p, oracle.q, oracle.C)  # type: ignore
        return dist / len(self.F._oracles)

    def composite_mp(self, gamma: float, z_k: SpacePoint, G_z_k: SpacePoint) -> SpacePoint:
        v_t = init_space_point(len(z_k.p), len(z_k.x))
        v_t_next = init_space_point(len(z_k.p), len(z_k.x))
        eta = 1 / (2 * gamma * self.L_F1)

        for t in range(self.composite_max_iters):
            G_t = self.F1.G(v_t) + G_z_k
            v_t_next.x = z_k.x * np.power(np.exp(-gamma * eta * G_t.x) * v_t.x / z_k.x, 1 / (eta + 1))
            v_t_next.p = z_k.p * np.power(np.exp(-gamma * eta * G_t.p) * v_t.p / z_k.p, 1 / (eta + 1))
            v_t_next.u = 0.5 * (z_k.u + v_t.u - gamma * eta * G_t.u)
            v_t_next.v = 0.5 * (z_k.v + v_t.v - gamma * eta * G_t.v)
            v_t_next = project_onto_space(v_t_next)

            G_t_next = self.F1.G(v_t_next) + G_z_k
            v_t.x = z_k.x * np.power(np.exp(-gamma * eta * G_t_next.x) * v_t.x / z_k.x, 1 / (eta + 1))
            v_t.p = z_k.p * np.power(np.exp(-gamma * eta * G_t_next.p) * v_t.p / z_k.p, 1 / (eta + 1))
            v_t.u = 0.5 * (z_k.u + v_t.u - gamma * eta * G_t_next.u)
            v_t.v = 0.5 * (z_k.v + v_t.v - gamma * eta * G_t_next.v)
            v_t = project_onto_space(v_t)

        return v_t

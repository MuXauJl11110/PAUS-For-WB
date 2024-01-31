from collections import defaultdict

import numpy as np
import ot

from oracles import OperatorOracle
from oracles.point import HPointType
from utils.space import init_space_point, project_onto_space


class MirrorProx:
    def __init__(
        self,
        F: OperatorOracle,
        log: bool = True,
        bar_true: np.ndarray | None = None,
    ):
        """
        :param OperatorOracle F: Operator of the problem.
        :param bool log: Logging, defaults to False
        """
        self.F = F
        self.log = log

        self.bar_true = bar_true
        if self.bar_true is not None:
            self.dist_true = 0.0
            for oracle in self.F._oracles:
                self.dist_true += ot.emd2(bar_true, oracle.q, oracle.C)  # type: ignore
            self.dist_true /= len(self.F._oracles)

    def fit(
        self, L: float, gamma: float | None = None, max_iter: int = 1000
    ) -> tuple[HPointType, dict[str, list[float]]]:
        history: dict[str, list[float]] = defaultdict(list)

        z_k = init_space_point(len(self.F._grad_p), len(self.F._grad_x))
        z_k_next = init_space_point(len(self.F._grad_p), len(self.F._grad_x))

        output_p = z_k.p.copy()
        if gamma is None:
            gamma = 1 / L

        for i in range(max_iter):
            G = self.F.G(z_k)
            z_k_next.x = z_k.x * np.exp(-gamma * G.x)
            z_k_next.p = z_k.p * np.exp(-gamma * G.p)
            z_k_next.u = z_k.u - gamma * G.u
            z_k_next.v = z_k.v - gamma * G.v
            z_k_next = project_onto_space(z_k)

            G_next = self.F.G(z_k_next)
            z_k.x = z_k.x * np.exp(-gamma * G_next.x)
            z_k.p = z_k.p * np.exp(-gamma * G_next.p)
            z_k.u = z_k.u - gamma * G_next.u
            z_k.v = z_k.v - gamma * G_next.v
            z_k = project_onto_space(z_k)

            output_p = ((i + 1) * output_p + z_k.p) / (i + 2)
            if self.log and self.bar_true is not None and (i % 50 == 0):
                d_gap = self.dual_gap(output_p)
                print(f"Iter: {i}, Dual gap: {d_gap}")
                history["dual_gap"].append(d_gap)
                history["iter"].append(i)  # type: ignore

        return output_p, history

    def dual_gap(self, p: np.ndarray) -> float:
        dist = 0.0
        for oracle in self.F._oracles:
            dist += ot.emd2(p, oracle.q, oracle.C)  # type: ignore
        return dist / len(self.F._oracles) - self.dist_true

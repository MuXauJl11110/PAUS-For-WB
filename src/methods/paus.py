from collections import defaultdict

import ot
import torch

from src.oracles.oracle import OperatorOracle
from src.oracles.point import OperatorPoint, Point
from src.utils.space import init_space_point, project_onto_space


class PAUS:
    def __init__(
        self,
        F: OperatorOracle,
        F1: OperatorOracle,
        L_F1: float,
        composite_max_iters: int,
        log: bool = True,
        bar_true: torch.Tensor | None = None,
        device: int | None = None,
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
            for q_i in self.F._q:
                self.dist_true += ot.emd2(bar_true, q_i, self.F._C)
            self.dist_true /= len(self.F._q)
        self.device = "cpu" if device is None else f"cuda:{device}"

    def fit(
        self, delta: float, gamma: float | None = None, max_iter: int = 1000
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        history: dict[str, list[float]] = defaultdict(list)

        z_k = init_space_point(self.F._d, self.F._T, self.device)
        u_k = init_space_point(self.F._d, self.F._T, self.device)

        output_p = z_k.log_p.clone()
        if gamma is None:
            gamma = 1 / delta

        d_gap = self.dual_gap(output_p)
        print(f"Iter: 0, Dual gap: {d_gap}")
        history["dual_gap"].append(d_gap.item())
        history["iter"].append(i)
        for i in range(max_iter):
            G_z_k = self.F.G(z_k) - self.F1.G(z_k)
            u_k = self.composite_mp(gamma, z_k, G_z_k)

            G_u_k = self.F.G(u_k) - self.F1.G(u_k)
            G = G_u_k - G_z_k
            z_k.log_x = u_k.log_x - gamma * G.x
            z_k.log_p = u_k.log_p - gamma * G.p
            z_k.u = u_k.u - gamma * G.u
            z_k.v = u_k.v - gamma * G.v
            z_k = project_onto_space(z_k)
            output_p = (i * output_p + torch.exp(u_k.log_p)) / (i + 1)
            output_p /= output_p.sum()
            if self.log and self.bar_true is not None and (i % 1 == 0):
                d_gap = self.dual_gap(output_p)
                print(f"Iter: {i+1}, Dual gap: {d_gap}")
                history["dual_gap"].append(d_gap.item())
                history["iter"].append(i)

        return output_p, history

    def composite_mp(self, gamma: float, z_k: Point, G_z_k: OperatorPoint) -> Point:
        v_t = init_space_point(self.F._d, self.F._T, self.device)
        v_t_next = init_space_point(self.F._d, self.F._T, self.device)
        eta = 1 / (2 * gamma * self.L_F1)

        for t in range(self.composite_max_iters):
            G_t = self.F1.G(v_t) + G_z_k
            v_t.log_x = (eta * z_k.log_x + v_t.log_x - gamma * eta * G_t.x) / (eta + 1)
            v_t.log_p = (eta * z_k.log_p + v_t.log_p - gamma * eta * G_t.p) / (eta + 1)
            v_t_next.u = 0.5 * (z_k.u + v_t.u - gamma * eta * G_t.u)
            v_t_next.v = 0.5 * (z_k.v + v_t.v - gamma * eta * G_t.v)
            v_t_next = project_onto_space(v_t_next)

            G_t_next = self.F1.G(v_t_next) + G_z_k
            v_t.log_x = (eta * z_k.log_x + v_t.log_x - gamma * eta * G_t_next.x) / (eta + 1)
            v_t.log_p = (eta * z_k.log_p + v_t.log_p - gamma * eta * G_t_next.p) / (eta + 1)
            v_t.u = 0.5 * (z_k.u + v_t.u - gamma * eta * G_t_next.u)
            v_t.v = 0.5 * (z_k.v + v_t.v - gamma * eta * G_t_next.v)
            v_t = project_onto_space(v_t)

        return v_t

    def dual_gap(self, p: torch.Tensor) -> float:
        dist = 0.0
        for q_i in self.F._q:
            dist += ot.emd2(p, q_i, self.F._C)
        return dist / self.F._d - self.dist_true

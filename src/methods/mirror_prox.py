from collections import defaultdict

import ot
import torch

from src.oracles.oracle import OperatorOracle
from src.oracles.point import OperatorPoint
from src.utils.space import init_space_point, project_onto_space


class MirrorProx:
    def __init__(
        self,
        F: OperatorOracle,
        log: bool = True,
        bar_true: torch.Tensor | None = None,
        device: int | None = None,
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
            for q_i in self.F._q:
                self.dist_true += ot.emd2(bar_true, q_i, self.F._C)
            self.dist_true /= len(self.F._q)
        self.device = "cpu" if device is None else f"cuda:{device}"

    def fit(
        self, L: float, gamma: float | None = None, max_iter: int = 1000
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        history: dict[str, list[float]] = defaultdict(list)

        z_k = init_space_point(self.F._d, self.F._T, self.device)
        z_k_next = init_space_point(self.F._d, self.F._T, self.device)

        output_p = torch.zeros_like(z_k.log_p)
        if gamma is None:
            gamma = 1 / L

        for i in range(max_iter):
            G: OperatorPoint = self.F.G(z_k)
            z_k_next.log_x = z_k.log_x - gamma * G.x
            z_k_next.log_p = z_k.log_p - gamma * G.p
            z_k_next.u = z_k.u - gamma * G.u
            z_k_next.v = z_k.v - gamma * G.v
            z_k_next = project_onto_space(z_k)

            G_next = self.F.G(z_k_next)
            z_k.log_x = z_k.log_x - gamma * G_next.x
            z_k.log_p = z_k.log_p - gamma * G_next.p
            z_k.u = z_k.u - gamma * G_next.u
            z_k.v = z_k.v - gamma * G_next.v
            z_k = project_onto_space(z_k)

            output_p = (i * output_p + torch.exp(z_k.log_p)) / (i + 1)
            output_p /= output_p.sum()
            if self.log and self.bar_true is not None and (i % 10 == 0):
                d_gap = self.dual_gap(output_p)
                print(f"Iter: {i}, Dual gap: {d_gap}")
                history["dual_gap"].append(d_gap.item())
                history["iter"].append(i)

        return output_p, history

    def dual_gap(self, p: torch.Tensor) -> float:
        dist = 0.0
        for q_i in self.F._q:
            dist += ot.emd2(p, q_i, self.F._C)
        return dist / self.F._d - self.dist_true

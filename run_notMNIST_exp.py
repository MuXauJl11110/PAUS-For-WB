import pickle

import click
import numpy as np

from methods.paus import PAUS
from oracles.oracle import OperatorOracle
from oracles.point import OTProblemOracle
from utils.dataset import load_notMNIST_small
from utils.tm import get_2d_tm


@click.command()
@click.option("--t", default=50, help="Number of images.")
@click.option("--n", default=30, help="Number of images on the server.")
@click.option(
    "--method_name",
    default="paus",
    help="Method name.",
    type=click.Choice(["paus", "mirror-prox"], case_sensitive=False),
)
@click.option("--step_scale", default=1.5, help="Step scale.")
@click.option("--max_iter", default=10000, help="Max iter.")
@click.option("--composite_max_iter", default=10, help="Composite max iterations.")
def run_notMNIST(d: int, t: int, n: int, method_name: str, step_scale: float, max_iter: int, composite_max_iter: int):
    data = load_notMNIST_small("B")
    C = get_2d_tm(28)
    oracles = [OTProblemOracle(data[i], C) for i in range(t)]
    F, F1 = OperatorOracle(oracles, d, t), OperatorOracle(oracles, d, n)
    if method_name == "paus":
        delta = 2 * (t - n) / t
        gamma = 1 / delta * step_scale
        method = PAUS(F, F1, 2, composite_max_iter, True)
        p_star, history = method.fit(
            delta,
            gamma=gamma,
            max_iter=max_iter,
        )
        method_name += str(composite_max_iter)
    else:
        raise ValueError("Unknown method!")
    with open(
        f"reports/notMNIST/{method_name}_history_d={d}_T={t}_n={n}_step_scale={step_scale}_max_iter={max_iter}.pkl",
        "wb",
    ) as f:
        pickle.dump(history, f)
    with open(
        f"reports/notMNIST/{method_name}_p_star_d={d}_T={t}_n={n}_step_scale={step_scale}_max_iter={max_iter}.pkl",
        "wb",
    ) as f:
        pickle.dump(p_star, f)


if __name__ == "__main__":
    np.random.seed(30)
    run_notMNIST()

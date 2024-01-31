import pickle

import click
import numpy as np

from methods.paus import PAUS


@click.command()
@click.option("--d", default=100, help="Support len.")
@click.option("--t", default=10, help="Number of gaussians.")
@click.option("--n", default=5, help="Number of gaussians on the server.")
@click.option(
    "--method_name",
    default="paus",
    help="Method name.",
    type=click.Choice(["paus", "mirror-prox"], case_sensitive=False),
)
@click.option("--step_scale", default=1.0, help="Step scale.")
@click.option("--max_iter", default=10000, help="Max iter.")
@click.option("--composite_max_iter", default=100, help="Composite max iterations.")
def run_gaus(d: int, t: int, n: int, method_name: str, step_scale: float, max_iter: int, composite_max_iter: int):
    with open(f"reports/oracles/F_d={d}_T={t}_n={n}.pkl", "rb") as f:
        F = pickle.load(f)
    with open(f"reports/oracles/F1_d={d}_T={t}_n={n}.pkl", "rb") as f:
        F1 = pickle.load(f)
    with open(f"reports/oracles/bar_true_d={d}_T={t}_n={n}.pkl", "rb") as f:
        bar_true = pickle.load(f)
    if method_name == "paus":
        delta = 2 * (t - n) / t
        gamma = 1 / delta * step_scale
        method = PAUS(F, F1, 2, composite_max_iter, True, bar_true=bar_true)
        p_star, history = method.fit(
            delta,
            gamma=gamma,
            max_iter=max_iter,
        )
        method_name += str(composite_max_iter)
    else:
        raise ValueError("Unknown method!")
    with open(
        f"reports/methods/{method_name}_history_d={d}_T={t}_n={n}_step_scale={step_scale}_max_iter={max_iter}.pkl",
        "wb",
    ) as f:
        pickle.dump(history, f)
    with open(
        f"reports/methods/{method_name}_p_star_d={d}_T={t}_n={n}_step_scale={step_scale}_max_iter={max_iter}.pkl", "wb"
    ) as f:
        pickle.dump(p_star, f)


if __name__ == "__main__":
    np.random.seed(30)
    run_gaus()

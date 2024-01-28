import pickle

import click
import numpy as np

from utils.experiment import get_method_gaussians, get_method_mnist


@click.command()
@click.option("--step_scale", default=1.0, help="Step scale.")
def run_gaus(step_scale: float):
    method, delta = get_method_gaussians(100, 100)
    gamma = 1 / delta * step_scale
    z_star, history = method.fit(delta, gamma=gamma)
    with open(f"paus_gaussian_history_{step_scale}.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(f"paus_gaussian_z_{step_scale}.pkl", "wb") as f:
        pickle.dump(z_star, f)


if __name__ == "__main__":
    np.random.seed(30)
    run_gaus()

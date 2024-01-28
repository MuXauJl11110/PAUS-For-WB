import pickle

import click
import numpy as np

from utils.experiment import get_method_mnist


@click.command()
@click.option("--target_digit", default=3, help="Number of greetings.")
def run_mnist(target_digit: int):
    method, delta = get_method_mnist(target_digit, 50)
    z_star, history = method.fit(delta)
    with open(f"paus_mnist_history_{target_digit}.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(f"paus_mnist_z_{target_digit}.pkl", "wb") as f:
        pickle.dump(z_star, f)


if __name__ == "__main__":
    np.random.seed(30)
    run_mnist()

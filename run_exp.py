import pickle

import numpy as np

from utils.experiment import get_method_gaussians, get_method_mnist

if __name__ == "__main__":
    np.random.seed(30)

    method, delta = get_method_gaussians(100, 100)
    z_star, history = method.fit(delta)
    with open(f"paus_gaussian_history.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(f"paus_gaussian_z.pkl", "wb") as f:
        pickle.dump(z_star, f)

    method, delta = get_method_mnist(3, 50)
    z_star, history = method.fit(delta)
    with open(f"paus_mnist_history.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(f"paus_mnist_z.pkl", "wb") as f:
        pickle.dump(z_star, f)

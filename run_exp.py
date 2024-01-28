import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from methods.paus import PAUS
from oracles.oracle import OperatorOracle
from oracles.point import OTProblemOracle
from utils.dataset import get_gaussian, load_mnist64, load_mnist784
from utils.tm import get_2d_tm

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "10"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "10"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "10"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "10"  # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = "10"  # export NUMEXPR_NUM_THREADS=1

    np.random.seed(30)

    d = 100
    digits, bar_true = get_gaussian(d, 100)
    C = get_2d_tm(int(np.sqrt(d)))
    T = 100  # number of all
    assert T <= len(digits)
    n = 50
    assert n <= T
    random_digits = digits[np.random.choice(len(digits), size=T, replace=False), :]
    oracles = [OTProblemOracle(random_digits[i], C) for i in range(T)]
    F1 = OperatorOracle(oracles, d, n)
    F = OperatorOracle(oracles, d, T)
    method = PAUS(F, F1, 1, 1000, True, bar_true=bar_true)
    delta = 2 * (T - n) / T
    z_star, history = method.fit(delta)
    with open(f"paus_gaussian_history.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(f"paus_gaussian_z.pkl", "wb") as f:
        pickle.dump(z_star, f)
    with open(f"paus_gaussian_digits.pkl", "wb") as f:
        pickle.dump(random_digits, f)
    with open(f"paus_gaussian_bartrue.pkl", "wb") as f:
        pickle.dump(bar_true, f)

    digits = load_mnist784(4)
    d = len(digits[0])
    C = get_2d_tm(int(np.sqrt(d)))
    T = 100  # number of all
    assert T <= len(digits)
    n = 50
    assert n <= T
    random_digits = digits[np.random.choice(len(digits), size=T, replace=False), :]
    oracles = [OTProblemOracle(random_digits[i], C) for i in range(T)]
    F1 = OperatorOracle(oracles, d, n)
    F = OperatorOracle(oracles, d, T)
    method = PAUS(F, F1, 1, 1000, True)
    delta = 2 * (T - n) / T
    z_star, history = method.fit(delta)
    with open(f"paus_mnist_history.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(f"paus_mnist_z.pkl", "wb") as f:
        pickle.dump(z_star, f)
    with open(f"paus_mnist_digits.pkl", "wb") as f:
        pickle.dump(random_digits, f)

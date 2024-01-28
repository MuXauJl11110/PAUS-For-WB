import pickle

import numpy as np

from methods.paus import PAUS
from oracles.oracle import OperatorOracle
from oracles.point import OTProblemOracle
from utils.dataset import get_gaussian, load_mnist784
from utils.tm import get_1d_tm, get_2d_tm


def get_method_gaussians(d: int, T: int) -> tuple[PAUS, float]:
    C = get_1d_tm(d)
    histograms, bar_true = get_gaussian(d, T)
    n = T // 4 * 3
    delta = 2 * (T - n) / T
    oracles = [OTProblemOracle(histograms[i], C) for i in range(T)]
    F, F1 = OperatorOracle(oracles, d, T), OperatorOracle(oracles, d, n)
    method = PAUS(F, F1, 1, 100, True, bar_true=bar_true)

    with open(f"paus_gaussian_digits.pkl", "wb") as f:
        pickle.dump(histograms, f)
    with open(f"paus_gaussian_bartrue.pkl", "wb") as f:
        pickle.dump(bar_true, f)

    return method, delta


def get_method_mnist(target_num: int, T: int) -> tuple[PAUS, float]:
    assert 0 <= target_num <= 9
    data = load_mnist784(target_num)
    assert T <= len(data)
    d = 784
    n = T // 4 * 3
    delta = 2 * (T - n) / T
    C = get_2d_tm(28)
    digits = data[np.random.choice(len(data), size=T, replace=False), :]

    oracles = [OTProblemOracle(digits[i], C) for i in range(T)]
    F, F1 = OperatorOracle(oracles, d, T), OperatorOracle(oracles, d, n)
    method = PAUS(F, F1, 1, 100, True)

    with open(f"paus_mnist_digits.pkl", "wb") as f:
        pickle.dump(digits, f)

    return method, delta

import os
from typing import Tuple

import matplotlib.image as mpimg
import numpy as np
from sklearn.datasets import fetch_openml, load_digits

from utils.space import project_onto_simplex


def get_gaussian(
    d: int, K: int, mu_min: float = -5.0, mu_max: float = 5.0, sigma_min: float = 0.8, sigma_max: float = 1.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param int d: Gaussian support size.
    :param int K: Number of distributions.
    :param float mu_min: Minimum expectation., defaults to -5.0
    :param float mu_max: Maximum expectation, defaults to 5.0
    :param float sigma_min: TBD., defaults to 0.8
    :param float sigma_max: TBD., defaults to 1.8
    :return Tuple[np.ndarray, np.ndarray]: TBD.
    """

    def gaussian(mu: np.ndarray, sigma: np.ndarray, z: np.ndarray) -> np.ndarray:
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((z - mu) ** 2) / (2 * sigma**2))

    z = np.linspace(-10, 10, d)
    gaus = np.zeros((K, d))
    mu = np.random.uniform(mu_min, mu_max, K)
    sigma = np.random.uniform(sigma_min, sigma_max, K)

    for i in range(K):
        gaus[i] = gaussian(mu[i], sigma[i], z)
        gaus[i] = project_onto_simplex(gaus[i])

    barmu = np.sum(mu) / len(mu)  # type: ignore
    barsigma = (np.sum(np.sqrt(sigma)) / len(sigma)) ** 2
    bar_true = project_onto_simplex(gaussian(barmu, barsigma, z))

    # return gaus, bar_true, z
    return gaus, bar_true  # type: ignore


def load_mnist784(target_digit: int):
    mndata = fetch_openml("mnist_784", version=1, as_frame=False, cache=True, data_home="datasets/MNIST")
    images, labels = mndata["data"], mndata["target"]
    normalized_digits = []

    for digit, label in zip(images, map(int, labels)):
        if label == target_digit:
            digit = np.array(digit, dtype=np.float64)
            normalized_digits.append(project_onto_simplex(digit))

    return np.array(normalized_digits)


def load_notMNIST_small(target_character: str):
    dataset = []
    for path in os.listdir(f"./datasets/notMNIST_small/{target_character}"):
        path_letter = os.path.join(f"./datasets/notMNIST_small/{target_character}", path)
        try:
            img = mpimg.imread(path_letter)
            img_n = project_onto_simplex(img)
            dataset.append(img_n.flatten())
        except:
            pass

    return np.array(dataset)


def load_mnist64(target_digit: int):
    digits = load_digits()
    normalized_digits = []

    for i, digit in enumerate(digits.images):  # type: ignore
        if digits.target[i] == target_digit:  # type: ignore
            normalized_digits.append(project_onto_simplex(digit).flatten())

    return np.array(normalized_digits)

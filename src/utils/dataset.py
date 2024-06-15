import os
from typing import Tuple

import matplotlib.image as mpimg
import numpy as np
import torch
from sklearn.datasets import fetch_openml, load_digits


def get_gaussian(
    d: int, K: int, mu_min: float = -5.0, mu_max: float = 5.0, sigma_min: float = 0.8, sigma_max: float = 1.8
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        gaus[i] = gaus[i] / sum(gaus[i])

    barmu = np.sum(mu) / len(mu)  # type: ignore
    barsigma = (np.sum(np.sqrt(sigma)) / len(sigma)) ** 2
    bar_true = gaussian(barmu, barsigma, z)
    bar_true /= bar_true.sum()

    # return gaus, bar_true, z
    return torch.from_numpy(gaus), torch.from_numpy(bar_true)  # type: ignore


def load_mnist784(target_digit: int) -> torch.Tensor:
    mndata = fetch_openml("mnist_784", version=1, as_frame=False, cache=True, data_home="datasets/MNIST")
    images, labels = mndata["data"], mndata["target"]
    normalized_digits = []

    for digit, label in zip(images, map(int, labels)):
        if label == target_digit:
            digit = np.array(digit, dtype=np.float64)
            normalized_digits.append(digit / digit.sum())

    return torch.from_numpy(np.array(normalized_digits))


def load_notMNIST_small(target_character: str) -> torch.Tensor:
    dataset = []
    for path in os.listdir(f"./datasets/notMNIST_small/{target_character}"):
        path_letter = os.path.join(f"./datasets/notMNIST_small/{target_character}", path)
        try:
            img = mpimg.imread(path_letter)
            img_n = img / img.sum()
            dataset.append(img_n.flatten())
        except:
            pass

    return torch.from_numpy(np.array(dataset))


def load_mnist64(target_digit: int) -> torch.Tensor:
    digits = load_digits()
    normalized_digits = []

    for i, digit in enumerate(digits.images):  # type: ignore
        if digits.target[i] == target_digit:  # type: ignore
            normalized_digits.append((digit / digit.sum()).flatten())

    return torch.from_numpy(np.array(normalized_digits, dtype=np.float32))

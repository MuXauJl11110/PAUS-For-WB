import numpy as np
from tqdm.notebook import tnrange

from oracles.space import SpacePoint


def init_space_point(d: int, T: int) -> SpacePoint:
    x = np.array([np.ones((d, d)) / (d * d)] * T)
    p = np.ones(d) / d
    u = np.array([np.zeros(d)] * T)
    v = np.array([np.zeros(d)] * T)

    return SpacePoint(x, p, u, v)


def project_onto_simplex(x: np.ndarray, eps: float = 0.00001) -> np.ndarray:
    x[x == 0] = eps
    x /= np.abs(x).sum()
    return x


def project_onto_inf_ball(x: np.ndarray, eps: float = 0.00001, radius: float = 1.0) -> np.ndarray:
    mask = np.abs(x) > radius
    x[mask] = np.sign(x[mask])
    return x


def project_onto_space(z: SpacePoint) -> SpacePoint:
    T = len(z.x)
    assert len(z.u) == T
    assert len(z.v) == T

    for i in tnrange(T):  # , desc="Computing projection onto X"):
        z.x[i] = project_onto_simplex(z.x[i])
    z.p = project_onto_simplex(z.p)

    for i in tnrange(T):  # , desc="Computing projection onto U"):
        z.u[i] = project_onto_inf_ball(z.u[i])

    for i in tnrange(T):  # , desc="Computing projection onto V"):
        z.v[i] = project_onto_inf_ball(z.v[i])

    return z

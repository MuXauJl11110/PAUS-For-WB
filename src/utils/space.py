import torch

from src.oracles.point import Point


def init_space_point(d: int, T: int) -> Point:
    log_x = torch.log(torch.ones(T, d, d) / (d * d))
    log_p = torch.log(torch.ones(d) / d)
    u = torch.zeros(T, d)
    v = torch.zeros(T, d)

    return Point(log_x, log_p, u, v)


def project_onto_space(z: Point, radius: float = 1.0) -> Point:
    x, p, u, v = torch.exp(z.log_x), torch.exp(z.log_p), z.u, z.v

    x /= x.sum(dim=(1, 2))[:, None, None]
    p /= p.sum()

    mask_u, mask_v = torch.abs(u) > radius, torch.abs(v) > radius
    u[mask_u] = torch.sign(u[mask_u])
    v[mask_v] = torch.sign(v[mask_v])

    return Point(torch.log(x), torch.log(p), u, v)

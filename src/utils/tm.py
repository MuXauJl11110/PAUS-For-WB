import ot
import torch


def get_1d_tm(d: int, device: int | None = None) -> torch.Tensor:
    transport_matrix = ot.utils.dist0(d)
    transport_matrix /= transport_matrix.max()  # type: ignore
    device = "cpu" if device is None else f"cuda:{device}"
    return torch.from_numpy(transport_matrix).float().to(device)


def get_2d_tm(d: int, device: int | None = None) -> torch.Tensor:
    tm_height = d**2
    tm_width = d**2
    device = "cpu" if device is None else f"cuda:{device}"
    transport_matrix = torch.zeros((tm_height, tm_width), device=device)

    for i in range(tm_height):
        for j in range(tm_width):
            x_i = i % d
            y_i = i // d
            x_j = j % d
            y_j = j // d
            transport_matrix[i][j] = (x_i - x_j) ** 2 + (y_i - y_j) ** 2
    transport_matrix /= transport_matrix.max()
    return transport_matrix

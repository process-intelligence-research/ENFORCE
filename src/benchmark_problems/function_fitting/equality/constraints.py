import torch

def constraint_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y1 = y[:, 0]
    y2 = y[:, 1]
    x = x.squeeze(1)
    c1 = (0.5*y1)**2 + x**2 + y2
    return c1

# def constraint_function(x, y):
#     y1 = y[:, 0]
#     y2 = y[:, 1]
#     c1 = y1 ** 2 + y2 ** 2 - 1
#     return c1

# def constraint_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     y1 = y[:, 0]
#     y2 = y[:, 1]
#     # x = x.squeeze(1)
#     c1 = y1**2 - y2**3
#     return c1


# def constraint_function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     y1 = y[:, 0]
#     y2 = y[:, 1]
#     x = x.squeeze(1)
#     c1 = y1 - y2**3 - 12*x**2 + 6*x - 6
#     return c1


def get_constraints():
    return constraint_function


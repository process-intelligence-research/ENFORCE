import torch

def constraints_column(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    
    y1 = y[:, 0]
    y2 = y[:, 1]
    y3 = y[:, 2]
    y4 = y[:, 3]
    y5 = y[:, 4]
    y6 = y[:, 5]
    y7 = y[:, 6]
    y8 = y[:, 7]
    y9 = y[:, 8]

    c1 = x1 + x2 - y1 - y2
    c2 = 0.697616946*x1 - y1*y3 - y2*y6
    c3 = 0.302383054*x1 - y1*y4 - y2*y7
    c4 = 1 - y3 - y4 - y5
    c5 = 1 - y6 - y7 - y8
    c6 = x3*y1 - y9
    
    return c1, c2, c3, c4, c5, c6
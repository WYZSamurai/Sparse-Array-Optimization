import torch


# NP种群数，Ny阵元y方向数，Nz阵元z方向数，NE=Ny*Nz为实际阵元数>=4，L口径y，H口径z，dc孔径约束
def gen(NP: int, Ny: int, Nz: int, L: float, H: float, dc: float):
    # (NP,Ny,Nz)
    y = torch.rand(NP, Ny, Nz)*(L-(Ny-1)*dc)
    y, _ = y.sort(1)
    y[:, 0, 0] = 0
    y[:, 0, -1] = 0
    y[:, -1, 0] = L-(Ny-1)*dc
    y[:, -1, -1] = L-(Ny-1)*dc
    dy = y+dc*torch.arange(0, Ny).unsqueeze(0).unsqueeze(-1)

    z = torch.rand(NP, Ny, Nz)*(H-(Nz-1)*dc)
    z, _ = z.sort(2)
    z[:, 0, 0] = 0
    z[:, -1, 0] = 0
    z[:, 0, -1] = H-(Nz-1)*dc
    z[:, -1, -1] = H-(Nz-1)*dc
    dz = z+dc*torch.arange(0, Nz).unsqueeze(0).unsqueeze(0)

    outputs = torch.complex(dy, dz)
    return outputs

import torch


def gen(NP: int, Ny: int, Nz: int, L: float, H: float, dc: float, device: torch.device):
    """
    生成初始种群

    NP种群数，Ny阵元y方向数，Nz阵元z方向数，NE=Ny*Nz为实际阵元数>=4，L口径y，H口径z，dc孔径约束
    输出dd真实距离(加入dc) d调整值(没加入dc)
    """
    assert H-(Nz-1)*dc >= 0 or L-(Ny-1)*dc >= 0, "警告！！！！！阵元间距小于限制值。\n" + \
        "H-(Nz-1)*dc="+str(H-(Nz-1)*dc)+"\n" + \
        "L-(Ny-1)*dc="+str(L-(Ny-1)*dc)+"\n"

    # (NP,Ny,Nz)
    y = torch.rand(NP, Ny, Nz, device=device)*(L-(Ny-1)*dc)
    y, _ = y.sort(1)
    y[:, 0, 0] = 0
    y[:, 0, -1] = 0
    y[:, -1, 0] = L-(Ny-1)*dc
    y[:, -1, -1] = L-(Ny-1)*dc

    z = torch.rand(NP, Ny, Nz, device=device)*(H-(Nz-1)*dc)
    z, _ = z.sort(2)
    z[:, 0, 0] = 0
    z[:, -1, 0] = 0
    z[:, 0, -1] = H-(Nz-1)*dc
    z[:, -1, -1] = H-(Nz-1)*dc

    dy = y+dc*torch.arange(0, Ny, device=device).unsqueeze(0).unsqueeze(-1)
    dz = z+dc*torch.arange(0, Nz, device=device).unsqueeze(0).unsqueeze(0)

    dd = torch.complex(dy, dz)
    d = torch.complex(y, z)
    return dd, d

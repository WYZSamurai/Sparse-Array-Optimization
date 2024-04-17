import torch


# NP种群数，Ny阵元y方向数，Nz阵元z方向数，L口径y，H口径z，dc孔径约束
def gen(NP: int, Ny: int,  Nz: int, L: float, H: float, dc: float):
    # (NP,Ny,Nz)
    y = torch.rand(NP, Ny)*(L-(Ny-1)*dc)
    y, _ = y.sort(1)
    y[0] = 0
    y[-1] = L-(Ny-1)*dc
    print(y)

    z = torch.rand(NP, Nz)*(H-(Nz-1)*dc)
    z, _ = z.sort(1)
    z[0] = 0
    z[-1] = H-(Nz-1)*dc
    print(z)

    f = torch.complex(y.unsqueeze(-1), z.unsqueeze(1))
    print(f[0])
    d = torch.complex(dc*torch.arange(0, Ny).unsqueeze(0).unsqueeze(-1),
                      dc*torch.arange(0, Nz).unsqueeze(0).unsqueeze(0))
    ff = f+d
    print(ff.shape)


gen(3, 4, 5, 8, 10, 0.5)

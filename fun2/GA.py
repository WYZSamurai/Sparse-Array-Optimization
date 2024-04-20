import torch
import pattern


def fitness(mag: torch.Tensor, phase0: torch.Tensor, lamb: float, ff: torch.Tensor, theta0: float, phi0: float, dt: int, dp: int):
    # 适应值计算
    Fdb1, Fdb2 = pattern.pattern2d(mag, phase0, lamb, ff, theta0, phi0, dt, dp)
    fit = -(pattern.Sll(Fdb1)+pattern.Sll(Fdb2))
    maxidx = fit.argmax()
    minidx = fit.argmin()
    ffbest = ff[maxidx]
    if fit[maxidx]-fit[minidx] == 0:
        fit = fit/fit.sum()
    else:
        fit = (fit-fit[minidx])/(fit[maxidx]-fit[minidx])
    return fit, ffbest


def select(f: torch.Tensor, fit: torch.Tensor):
    # 选择阶段
    NP, _, _ = f.shape
    P = (fit/fit.sum())
    index = torch.multinomial(input=P, num_samples=NP, replacement=True)
    f = f[index]
    return f


def cross():
    # 交叉阶段
    pass


def mutation():
    # 变异阶段
    pass


def reform():
    pass


def GA(ff: torch.Tensor, f: torch.Tensor, lamb: float, theta0: float, phi0: float, dt: int, dp: int, G: int):
    NP, Ny, Nz = ff.shape

    mag = torch.ones(NP, Ny, Nz)
    phase0 = torch.zeros_like(mag)

    # (G,Ny,Nz)
    ffbest = torch.complex(torch.zeros(G, Ny, Nz), torch.zeros(G, Ny, Nz))

    # 主程序
    for i in range(G):
        fit, ffbest[i] = fitness(mag, phase0, lamb, ff, theta0, phi0, dt, dp)
        ff = select(ff, fit)

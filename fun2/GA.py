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


def cross(f: torch.Tensor, Pc: float) -> torch.Tensor:
    """
    对个体群体进行交叉（重组）
    参数:
        dna: 3D形状张量（NP，Ny，Nz），表示个体的群体，其中每个个体是形状矩阵（Ny，Nz.）。
        Pc: 表示交叉概率的0和1之间的浮点值。

    输出:
        将交叉的改良dna张量应用于选定的成对个体。
    """
    NP, Ny, Nz = f.shape
    P = torch.rand(NP,)-Pc
    for i in range(NP):
        if P[i] < 0:
            cuty = torch.randint(0, Ny, (1,))
            cutz = torch.randint(0, Nz, (1,))

    return f


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

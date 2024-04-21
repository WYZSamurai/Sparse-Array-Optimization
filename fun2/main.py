import torch
import generate
import GA


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# 天线相关
# y方向
L = 9.5
# z方向
H = 4.5
lamb = 1.0
dc = 0.5*lamb
theta0 = 0.0
phi0 = 0.0
Ny = 10
Nz = 10


# 算法相关
NP = 3
G = 1
Pc = 0.8
Pm = 0.050
dt = 360
dp = 360


# 生成种群(NP, Ny, Nz)
ff, f = generate.gen(NP, Ny, Nz, L, H, dc, device)
GA.GA(ff, f, lamb, theta0, phi0, dt, dp, G)

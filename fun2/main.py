import torch
import generate
import pattern


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# 天线相关
L = 9.5
H = 4.5
lamb = 1.0
dc = 0.5*lamb
theta0 = 0.0
phi0 = 0.0
Ny = 10
Nz = 10
# Ny = 3
# Nz = 3


# 算法相关
NP = 50
# NP = 3
G = 100
Pc = 0.8
Pm = 0.050
dt = 360
dp = 360


# 激励相位(NP, Ny, Nz)
mag = torch.ones(NP, Ny, Nz)
phase0 = torch.zeros_like(mag)


# 生成种群(NP, Ny, Nz)
ff, f = generate.gen(NP, Ny, Nz, L, H, dc)
print(ff[0])

# 绘制截面方向图
Fdb = pattern.patternt(mag[0], phase0[0], lamb, ff[0], theta0, phi0, dt)
pattern.plott(Fdb, dt)


# 绘制3d方向图
# Fdb = pattern.pattern(mag[0], phase0[0], lamb, ff[0], theta0, phi0, dt, dp)
# pattern.plot(Fdb, dt, dp)

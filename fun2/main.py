import torch
import generate
import GA
import time


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# 天线相关
# y方向
L = 20
# z方向
H = 20
lamb = 1.0
dc = 0.5*lamb
theta0 = 0.0
phi0 = 0.0
Ny = 16
Nz = 16


# 算法相关
NP = 1000
G = 500
Pc = 0.8
Pm = 0.100
dt = 360
dp = 360


# 生成种群(NP, Ny, Nz)
ff, f = generate.gen(NP, Ny, Nz, L, H, dc, device)
begin = time.time()
fitbest, ffbest = GA.GA(ff, f, lamb, theta0, phi0, dt, dp, G, Pc, Pm, L, H, dc)
end = time.time()
total = end-begin
print("算法总耗时：", total)


print("绘制最佳阵元位置图")
GA.pattern.poltff(ffbest)
print("完成")


print("绘制适应度曲线。")
GA.plot(G, fitbest)
print("完成")


print("绘制最佳个体的3d方向图")
mag = torch.ones(Ny, Nz)
phase0 = torch.zeros_like(mag)
Fdb = GA.pattern.pattern3d(mag, phase0, lamb, ffbest, theta0, phi0, dt, dp)
GA.pattern.plot3d(Fdb)
print("完成")
print("结束程序")

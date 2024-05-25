import torch
import pattern
import plotly.graph_objects as go


def fitness(mag: torch.Tensor, phase0: torch.Tensor, lamb: float, ff: torch.Tensor, theta0: float, phi0: float, dt: int, dp: int, f: torch.Tensor):
    """
    适应值计算
    """
    Fdb1, Fdb2 = pattern.pattern2d(mag, phase0, lamb, ff, theta0, phi0, dt, dp)
    sll1 = pattern.Sll(Fdb1)
    sll2 = pattern.Sll(Fdb2)
    print(sll1[0], sll2[0])
    fit = -(sll1+sll2)
    maxidx = fit.argmax()
    minidx = fit.argmin()
    fitbest = fit[maxidx]
    ffbest = ff[maxidx]
    fbest = f[maxidx]
    if fit[maxidx]-fit[minidx] == 0:
        fit = fit/fit.sum()
    else:
        fit = (fit-fit[minidx])/(fit[maxidx]-fit[minidx])

    print("此代最佳适应度为：", fitbest)
    return fit, ffbest, fbest, fitbest


def select(f: torch.Tensor, fit: torch.Tensor):
    """
    选择阶段
    """
    NP, _, _ = f.shape
    P = (fit/fit.sum())
    index = torch.multinomial(input=P, num_samples=NP, replacement=True)
    f = f[index]
    return f


def cross(f: torch.Tensor, Pc: float) -> torch.Tensor:
    """
    交叉阶段
    """
    NP, Ny, Nz = f.shape
    assert NP % 2 == 0, "NP应为偶数"
    P = torch.rand(NP,)-Pc
    for i in range(0, NP, 2):
        if P[i] < 0:
            cuty = torch.randint(0, Ny, (1,)).item()
            cutz = torch.randint(0, Nz, (1,)).item()
            temp = f[i, cuty:, cutz:].clone()
            f[i+1, cuty:, cutz:] = f[i, cuty:, cutz:]
            f[i, cuty:, cutz:] = temp
    return f


def mutation(f: torch.Tensor, Pm: float, L: float, H: float, dc: float):
    """
    变异阶段
    """
    NP, Ny, Nz = f.shape
    P = torch.rand(NP, Ny, Nz)-Pm
    # 列表中分别是NP，Ny，Nz的坐标
    idx = torch.where(P < 0)
    f[idx].real = torch.rand(f[idx].shape)*(L-(Ny-1)*dc)
    f[idx].imag = torch.rand(f[idx].shape)*(H-(Nz-1)*dc)
    return f


def reform(f: torch.Tensor, L: float, H: float, dc: float):
    """
    变形阶段
    """
    _, Ny, Nz = f.shape
    y, _ = f.real.sort(1)
    y[:, 0, 0] = 0
    y[:, 0, -1] = 0
    y[:, -1, 0] = L-(Ny-1)*dc
    y[:, -1, -1] = L-(Ny-1)*dc

    z, _ = f.imag.sort(2)
    z[:, 0, 0] = 0
    z[:, -1, 0] = 0
    z[:, 0, -1] = H-(Nz-1)*dc
    z[:, -1, -1] = H-(Nz-1)*dc

    dy = y+dc*torch.arange(0, Ny, device=f.device).unsqueeze(0).unsqueeze(-1)
    dz = z+dc*torch.arange(0, Nz, device=f.device).unsqueeze(0).unsqueeze(0)

    ff = torch.complex(dy, dz)
    f = torch.complex(y, z)
    return ff, f


def GA(ff: torch.Tensor, f: torch.Tensor, lamb: float, theta0: float, phi0: float, dt: int, dp: int, G: int, Pc: float, Pm: float, L: float, H: float, dc: float):
    """
    遗传算法主程序
    """
    NP, Ny, Nz = ff.shape

    mag = torch.ones(NP, Ny, Nz, device=ff.device)
    phase0 = torch.zeros_like(mag, device=ff.device)

    # 最佳位置(G,Ny,Nz)
    ffbest = torch.complex(torch.zeros(G, Ny, Nz), torch.zeros(G, Ny, Nz))
    # 最佳适应值
    fitbest = torch.zeros(G,)

    # 主程序
    print("开始优化")
    for i in range(G):
        print("第", i+1, "代")
        fit, ffbest[i], fbest, fitbest[i] = fitness(
            mag, phase0, lamb, ff, theta0, phi0, dt, dp, f)
        f = select(f, fit)
        f = cross(f, Pc)
        f = mutation(f, Pm, L, H, dc)
        ff, f = reform(f, L, H, dc)
        ff[0] = ffbest[i]
        f[0] = fbest
    print("算法结束")
    bestindex = fitbest.argmax()
    print("最佳适应值为：\n", fitbest[bestindex])
    print("最佳阵元位置为：\n", ffbest[bestindex])
    return fitbest, ffbest[bestindex]


def plot(G: int, ybest: torch.Tensor):
    """
    绘制适应度曲线
    """
    x = torch.linspace(1, G, G)
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=x,
            y=ybest,
        )
    )
    fig.update_layout(
        title="适应度曲线图",
        xaxis=dict(
            title="迭代次数",
        ),
        yaxis=dict(
            title="适应度",
        ),
        autosize=True,
        template="simple_white",
    )
    fig.show()

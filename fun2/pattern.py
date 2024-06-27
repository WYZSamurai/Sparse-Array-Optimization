import torch
import plotly.graph_objects as go


def pattern_multiple(mag: torch.Tensor, phase0: torch.Tensor, lamb: float, d: float, theta0: float, phi0: float, dt: int, dp: int):
    """
    多个个体的3d方向图
    """
    pi = torch.pi
    _, m, n = mag.shape
    k = 2*pi/lamb
    theta0 = torch.tensor(theta0) * pi / 180
    phi0 = torch.tensor(phi0) * pi / 180

    phi = torch.linspace(-pi/2, pi/2, dp, device=mag.device)
    theta = torch.linspace(-pi/2, pi/2, dt, device=mag.device)

    # 构造角度矩阵 (dt, dp)
    ang1 = torch.cos(theta.view(-1, 1)) * torch.sin(phi.view(1, -1)
                                                    ) - torch.cos(theta0) * torch.sin(phi0)
    ang2 = torch.sin(theta.view(-1, 1)) - torch.sin(theta0)

    # 构造距离矩阵 (m, n)
    dm = k * d * torch.arange(m, device=mag.device).view(m, 1)
    dn = k * d * torch.arange(n, device=mag.device).view(1, n)

    # 计算每个天线元的相位贡献(NP,m,n,dt,dp)
    phase_contributions = phase0.unsqueeze(3).unsqueeze(4) + dm.unsqueeze(2).unsqueeze(3).unsqueeze(0) * ang1.unsqueeze(
        0).unsqueeze(0).unsqueeze(0) + dn.unsqueeze(2).unsqueeze(3).unsqueeze(0) * ang2.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # 计算复数指数项并按天线元求和(NP,dt,dp)
    F = torch.abs(torch.sum(mag.unsqueeze(3).unsqueeze(4) * torch.exp(torch.complex(
        torch.zeros_like(phase_contributions), phase_contributions)), dim=(1, 2)))

    # 转换为分贝并进行归一化
    Fdb = 20 * torch.log10(F / torch.max(F))
    return Fdb


def pattern3d(mag: torch.Tensor, phase0: torch.Tensor, lamb: float, d: torch.Tensor, theta0: float, phi0: float, dt: int, dp: int):
    """
    单个个体的3d方向图
    mag\phase0:Ny*Nz
    d:真实位置信息（非间距）
    """
    pi = torch.pi
    k = 2 * pi / lamb

    theta0_rad = torch.tensor(theta0) * pi / 180
    phi0_rad = torch.tensor(phi0) * pi / 180

    theta = torch.linspace(-pi / 2, pi / 2, dt)
    phi = torch.linspace(-pi / 2, pi / 2, dp)

    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

    ang1 = torch.cos(theta_grid) * torch.sin(phi_grid) - \
        torch.cos(theta0_rad) * torch.sin(phi0_rad)
    ang2 = torch.sin(theta_grid) - torch.sin(theta0_rad)

    m, n = mag.shape
    dm = k * d.real.reshape(m, n, 1, 1) * ang1.reshape(1, 1, dt, dp)
    dn = k * d.imag.reshape(m, n, 1, 1) * ang2.reshape(1, 1, dt, dp)

    phase_contributions = phase0.unsqueeze(-1).unsqueeze(-1) + dm + dn
    complex_exponentials = torch.exp(torch.complex(
        torch.zeros_like(phase_contributions), phase_contributions))

    F = (mag.unsqueeze(-1).unsqueeze(-1) *
         complex_exponentials).sum(dim=(0, 1)).abs()
    Fdb = 20 * torch.log10(F / F.max()+0.0001)

    return Fdb


def pattern2d(mag: torch.Tensor, phase0: torch.Tensor, lamb: float, d: torch.Tensor, theta0: float, phi0: float, dt: int, dp: int):
    """
    群体theta截面的方向图
    """
    # d/mag/phase0 (NP,M,N)
    NP, m, n = mag.shape
    pi = torch.pi
    k = 2 * pi / lamb
    theta0_rad = torch.tensor(theta0) * pi / 180
    phi0_rad = torch.tensor(phi0) * pi / 180

    # (dt,)
    theta_rad = torch.linspace(-pi/2, pi/2, dt, device=d.device)
    # (dp,)
    phi_rad = torch.linspace(-pi/2, pi/2, dp, device=d.device)

    # (dt,)
    ang1 = torch.cos(theta_rad)*torch.sin(phi0_rad) - \
        torch.cos(theta0_rad)*torch.sin(phi0_rad)
    # (dt,)
    ang2 = torch.sin(theta_rad)-torch.sin(theta0_rad)
    # (NP,m,n,dt)
    dm = k*d.real.reshape(NP, m, n, 1)*ang1.reshape(1, 1, 1, dt)
    dn = k*d.imag.reshape(NP, m, n, 1)*ang2.reshape(1, 1, 1, dt)
    phase_contributions = phase0.unsqueeze(-1)+dm+dn
    complex_exponentials = torch.exp(torch.complex(
        torch.zeros_like(phase_contributions), phase_contributions))
    # (NP,dt)
    F = (mag.unsqueeze(-1) * complex_exponentials).sum(dim=(1, 2)).abs()
    Fdb1 = 20 * torch.log10(F / F.max()+0.0001)

    # (dp,)
    ang1 = torch.cos(theta0_rad)*torch.sin(phi_rad) - \
        torch.cos(theta0_rad)*torch.sin(phi0_rad)
    ang2 = torch.sin(theta0_rad)-torch.sin(theta0_rad)
    # (NP,m,n,dp)
    dm = k*d.real.reshape(NP, m, n, 1)*ang1.reshape(1, 1, 1, dp)
    dn = k*d.imag.reshape(NP, m, n, 1)*ang2
    phase_contributions = phase0.unsqueeze(-1)+dm+dn
    complex_exponentials = torch.exp(torch.complex(
        torch.zeros_like(phase_contributions), phase_contributions))
    # (NP,dp)
    F = (mag.unsqueeze(-1) * complex_exponentials).sum(dim=(1, 2)).abs()
    Fdb2 = 20 * torch.log10(F / F.max()+0.0001)

    return Fdb1, Fdb2


def Sll(Fdb: torch.Tensor):
    """
    输入种群的Fdb2d值,计算最大副瓣电平
    """
    batch_size, delta = Fdb.shape

    # 主瓣最大值的位置
    maxi = Fdb.argmax(dim=1)

    # 向右查找第一个电平增加的点
    diffs_right = torch.diff(Fdb, dim=1)
    right_changes = torch.cat([diffs_right, torch.full(
        (batch_size, 1), -float('inf'), device=Fdb.device)], dim=1) > 0

    # 构造一个大于maxi的序列索引矩阵
    idx_right = torch.arange(delta, device=Fdb.device).repeat(batch_size, 1)
    right_valid = (idx_right > maxi.unsqueeze(1)) & right_changes
    indexr = torch.where(right_valid, idx_right, delta)
    indexr = torch.min(indexr, dim=1).values - maxi

    # 向左查找第一个电平增加的点
    diffs_left = torch.diff(Fdb.flip(dims=[1]), dim=1)
    left_changes = torch.cat([diffs_left, torch.full(
        (batch_size, 1), -float('inf'), device=Fdb.device)], dim=1) > 0
    idx_left = torch.arange(delta, device=Fdb.device).repeat(batch_size, 1)
    left_valid = (idx_left > (delta - 1 - maxi).unsqueeze(1)) & left_changes
    indexl = torch.where(left_valid, idx_left, delta)
    # indexl(batch_size,)
    indexl = torch.min(indexl, dim=1).values - (delta - 1 - maxi)

    # 设置主瓣区域为-inf (maxi - indexl,maxi + indexr)
    mask = torch.ones_like(Fdb, dtype=torch.bool)
    ranges = torch.arange(delta, device=Fdb.device).repeat(batch_size, 1)
    # 副瓣区域
    mask &= ~((ranges >= (maxi - indexl).unsqueeze(1)) &
              (ranges <= (maxi + indexr).unsqueeze(1)))

    Fdb = Fdb.masked_fill(~mask, float('-inf'))

    # 寻找调整后的最大电平(batch_size,)
    MSLL = Fdb.max(dim=1).values

    return MSLL


def plot3d(Fdb: torch.Tensor):
    """
    绘制个体3d方向图
    """
    dt, dp = Fdb.shape
    phi = torch.linspace(-90.0, 90.0, dp)
    theta = torch.linspace(-90.0, 90.0, dt)

    torch.meshgrid(theta, phi, indexing='ij')
    fig = go.Figure(data=[go.Surface(z=Fdb.cpu(), x=theta, y=phi)])

    # 更新图表布局
    fig.update_layout(
        title="3D方向图",
        scene=dict(
            xaxis_title='theta',
            yaxis_title='phi',
            zaxis_title='Fdb'
        ),
        autosize=True,
        template="simple_white",
    )

    # 显示图表
    fig.show()


def plot2d(Fdb: torch.Tensor):
    """
    绘制个体截面图
    """
    d = Fdb.shape[0]
    ang = torch.linspace(-90.0, 90.0, d)
    fig = go.Figure(data=[go.Scatter(x=ang, y=Fdb.cpu())])
    fig.update_layout(
        title="截面方向图",
        scene=dict(
            xaxis_title='ang',
            yaxis_title='Fdb'
        ),
        template="simple_white",
        autosize=True,
    )
    fig.show()


def poltff(ff: torch.Tensor):
    """
    绘制个体的阵元位置
    """
    Ny, Nz = ff.shape
    fig = go.Figure()
    fig.add_traces(
        data=go.Scatter(
            x=ff.cpu().real.reshape(Ny*Nz,),
            y=ff.cpu().imag.reshape(Ny*Nz,),
            mode="markers"
        ),
    )
    fig.update_layout(
        title="阵元位置图",
        xaxis=dict(
            title="y方向",
        ),
        yaxis=dict(
            title="z方向",
        ),
        autosize=True,
        template="simple_white",
    )
    fig.show()

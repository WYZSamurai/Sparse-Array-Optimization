import plotly.graph_objects as go
import torch


# 等间距，每点d的激励相位已知
def pattern(mag: torch.Tensor, phase_0: torch.Tensor, lamb: float, d: float, delta: int, theta_0: float):
    # mag/phase_0(batch_size,m)
    m = mag.shape[1]
    pi = torch.pi
    k = 2 * pi / lamb

    theta_0_rad = torch.tensor(theta_0) * pi / 180

    # (delta,)
    theta = torch.linspace(-pi/2, pi/2, delta, device=mag.device)
    ang = (torch.sin(theta) - torch.sin(theta_0_rad)).to(mag.device)

    # nd(m,)
    dm = k * d * torch.arange(0, m, device=mag.device)

    # 计算每个批次所有m值和所有delta值的所有相位
    # 广播以创建相位值的（batch_size，m，delta）矩阵
    phase = phase_0.unsqueeze(-1) + dm.unsqueeze(0).unsqueeze(-1) * \
        ang.unsqueeze(0).unsqueeze(0)

    # 使用欧拉公式将相位转换为复数并求和
    # 广播mag到（batch_size，m，delta）
    complex_exponential = mag.unsqueeze(-1) * torch.exp(
        torch.complex(torch.zeros_like(phase), phase))
    # 沿m维度求和，取大小
    F = torch.sum(complex_exponential, dim=1).abs()

    # 转换为db，按批次中每个个体的最大值进行归一化
    Fdb = 20 * torch.log10(F / F.max(dim=1, keepdim=True).values+0.0001)

    return Fdb


def plot(Fdb: torch.Tensor, delta: int, theta_min: float, theta_max: float):
    theta = torch.linspace(theta_min, theta_max, delta)
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=theta,
            y=Fdb.to(device=torch.device("cpu")),
        )
    )
    fig.update_layout(
        template="simple_white",
        title="方向图",
        xaxis_title="theta",
        xaxis_range=[theta_min-10, theta_max+10],
        yaxis_title="Fdb",
        yaxis_range=[-60, 0.5],
    )
    fig.show()

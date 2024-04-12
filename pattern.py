import plotly.graph_objects as go
import torch
import generate


# 线阵，等间距，等幅同相
# dna(ME,) Fdb(delta,)
def pattern(dna: torch.Tensor, lamb: float, d: float, delta: int, theta_0: float):
    ME = dna.shape[0]
    k = 2*torch.pi/lamb
    theta_0 = torch.tensor(theta_0)*torch.pi/180
    theta = torch.linspace(-torch.pi/2, torch.pi/2, delta)

    # ex(ME,)
    ex = torch.complex(dna, torch.zeros(ME,))
    # phi(delta,ME)
    phi = (torch.sin(theta)-torch.sin(theta_0)).reshape(delta, 1).repeat(1, ME)
    # nd(delta,ME)
    nd = k*d*torch.arange(0, ME).reshape(1, ME).repeat(delta, 1)

    phi = torch.exp(torch.complex(torch.zeros(delta, ME), phi*nd))
    F = torch.matmul(phi, ex).abs()
    Fdb = 20*torch.log10(F/F.max())
    # print("Fdb为：\n", Fdb)
    return Fdb


def plot(Fdb: torch.Tensor, delta: int, theta_min: float, theta_max: float):
    theta = torch.linspace(theta_min, theta_max, delta)
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=theta,
            y=Fdb,
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


if __name__ == "__main__":
    # 种群数
    NP = 50
    # 交叉率
    Pc = 0.8
    # 变异率
    Pm = 0.050
    # 迭代次数
    G = 200
    # 实际阵元个数
    NE = 5
    # 满阵阵元个数
    ME = 10
    # 波长（米）
    lamb = 1
    # 阵列间距
    d = 0.5*lamb
    # 波束指向（角度）
    theta_0 = 0
    # 扫描精度
    delta = 180
    # 阵列口径（米）
    AA = d*(ME-1)

    dna = generate.gen(NP, ME, NE)
    Fdb = pattern(dna[0], lamb, d, delta, theta_0)

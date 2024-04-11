import plotly.graph_objects as go
import torch


# 线阵，等间距，等幅同相
# dna(1,1,L) Fdb(delta,)
def pattern(dna: torch.Tensor, l: float, d: float, delta: int, theta_0: float):
    M = dna.shape[2]
    ex = torch.complex(dna.reshape(M, 1), torch.zeros((M, 1)))
    k = 2*torch.pi/l
    theta_0 = torch.tensor(theta_0)*torch.pi/180
    theta = torch.linspace(-torch.pi/2, torch.pi/2, delta)

    phi = (torch.sin(theta)-torch.sin(theta_0)).reshape(delta, 1)
    phi = torch.matmul(phi, torch.ones(1, M))

    nd = k*d*torch.arange(0, M).reshape(1, M)
    nd = torch.matmul(torch.ones(delta, 1), nd)

    phi = torch.exp(torch.complex(torch.zeros((delta, M)), phi*nd))
    F = torch.matmul(phi, ex).abs().reshape(delta,)
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
    theta_min = -90.0
    theta_max = 90.0
    (l, delta, theta_0) = (1, 360, 0)
    d = l/2
    G = 5
    NP = 100
    m = 1
    n = 1
    L = 20
    Pc = 0.8
    Pm = 0.050
    dna = torch.randint(0, 2, (m, n, L)).to(dtype=torch.float)
    Fdb = pattern(dna, l, d, delta, theta_0)
    print(Fdb)

import plotly.graph_objects as go
import torch
import msll
import pattern

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
device = torch.device("cpu")
cpu = torch.device("cpu")


def decode(dna: torch.Tensor, l: float, d: float, delta: int, theta_0: float):
    NP = dna.shape[0]

    fit = torch.zeros(NP,).to(dtype=torch.float, device=device)
    for i in range(NP):
        Fdb: torch.Tensor = pattern.pattern(dna[i], l, d, delta, theta_0)
        fit[i] = -msll.msll(Fdb)

    maxindex = torch.argmax(fit)
    minindex = torch.argmin(fit)

    dnabest = dna[maxindex]
    ybest = -fit[maxindex]

    if fit[maxindex]-fit[minindex] == 0:
        fit = fit/fit.sum()
    else:
        fit = (fit-fit[minindex])/(fit[maxindex]-fit[minindex])

    print("此代最佳msll为：", ybest)
    return fit, ybest, dnabest


def selection(dna: torch.Tensor, fit: torch.Tensor):
    NP = dna.shape[0]
    P = (fit/fit.sum()).reshape(NP,)
    index = torch.multinomial(input=P, num_samples=NP, replacement=True)
    dna = dna[index]
    return dna


def crossover(dna: torch.Tensor, Pc: float):
    NP = dna.shape[0]
    L = dna.shape[3]
    for i in range(0, NP, 2):
        P = torch.rand(size=(1,)).item()
        if P < Pc:
            randcut = torch.randint(int(L/2), L, (1,)).item()
            temp = dna[i, :, :, randcut:].clone()
            dna[i, :, :, randcut:] = dna[i+1, :, :, randcut:]
            dna[i+1, :, :, randcut:] = temp
    return dna


def mutation(dna: torch.Tensor, Pm: float):
    P = (torch.rand(size=dna.shape)-Pm * torch.ones(size=dna.shape))
    dna = torch.where(P > 0, input=dna, other=1-dna)
    return dna


def GA(dna: torch.Tensor, L: int, m: int, n: int, G: int, Pc: float, Pm: float, l: float, d: float, delta: int, theta_0: float):
    ybest = torch.zeros((G,))
    dnabest = torch.zeros((G, m, n, L))

    for i in range(G):
        print("第", i+1, "代")
        fit, ybest[i], dnabest[i] = decode(dna, l, d, delta, theta_0)
        dna = selection(dna, fit)
        dna = crossover(dna, Pc)
        dna = mutation(dna, Pm)
        dna[0] = dnabest[i]

    bestindex = torch.argmin(ybest)
    print("算法结束")
    print("最佳dna值为：\n", dnabest[bestindex])
    print("最佳函数值为：\n", ybest[bestindex])
    return ybest, dnabest[bestindex]


def plot(G: int, ybest: torch.Tensor):
    x = torch.linspace(1, G, G)
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=x,
            y=ybest,
        )
    )
    fig.update_layout(
        template="simple_white",
    )
    fig.show()


if __name__ == "__main__":
    pass

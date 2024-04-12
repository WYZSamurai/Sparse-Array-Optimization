import plotly.graph_objects as go
import torch
import generate
import pattern
import msll


def fitness(dna: torch.Tensor, lamb: float, d: float, delta: int, theta_0: float):
    NP = dna.shape[0]
    fit = torch.zeros(NP,).to(dtype=torch.float)

    for i in range(NP):
        Fdb = pattern.pattern(dna[i], lamb, d, delta, theta_0)
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
    (NP, ME) = dna.shape
    for i in range(0, NP, 2):
        P = torch.rand(size=(1,)).item()
        if P < Pc:
            # [1,ME-2]
            randcut = torch.randint(1, ME-1, (1,)).item()
            temp = dna[i, randcut:].clone()
            dna[i, randcut:] = dna[i+1, randcut:]
            dna[i+1, randcut:] = temp
    return dna


def mutation(dna: torch.Tensor, Pm: float):
    # (1,ME-2)
    P = (torch.rand(dna.shape)-Pm * torch.ones(dna.shape))
    dna = torch.where(P > 0, input=dna, other=1-dna)
    dna[:, [0, -1]] = 1
    return dna


def judge(dna: torch.Tensor, NE: int):
    (NP, ME) = dna.shape
    # 求和
    temp = torch.matmul(dna, torch.ones(ME,))
    for i in range(NP):
        # 大于  1->0
        if temp[i] > NE:
            # 找出所有1的索引
            idx = torch.where(dna[i, 1:ME-1] == 1)[0]+1
            # 需要个数P=temp[i]-NE的1->0
            dna[i][idx[torch.multinomial(torch.ones(
                idx.shape), int((temp[i]-NE).item()), False)]] = 0
        # 小于  0->1
        elif temp[i] < NE:
            idx = torch.where(dna[i, 1:ME-1] == 0)[0]+1
            dna[i][idx[torch.multinomial(torch.ones(
                idx.shape), int((NE-temp[i]).item()), False)]] = 1
    return dna


def GA(dna: torch.Tensor, G: int, Pc: float, Pm: float, NE: int, lamb: float, d: float, delta: int, theta_0: float):
    ME = dna.shape[1]
    ybest = torch.zeros(G,)
    dnabest = torch.zeros(G, ME)

    for i in range(G):
        print("第", i+1, "代")
        fit, ybest[i], dnabest[i] = fitness(dna, lamb, d, delta, theta_0)
        dna = selection(dna, fit)
        dna = crossover(dna, Pc)
        dna = mutation(dna, Pm)
        dna = judge(dna, NE)
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


def main():
    # 种群数
    NP = 50
    # 交叉率
    Pc = 0.8
    # 变异率
    Pm = 0.050
    # 迭代次数
    G = 50
    # 实际阵元个数
    NE = 50
    # 满阵阵元个数
    ME = 100
    # 波长（米）
    lamb = 1
    # 阵列间距
    d = 0.5*lamb
    # 波束指向（角度）
    theta_0 = 0
    # 扫描精度
    delta = 1800
    # 阵列口径（米）
    # AA = d*(ME-1)

    # dna(NP,ME)
    dna = generate.gen(NP, ME, NE)
    GA(dna, G, Pc, Pm, NE, lamb, d, delta, theta_0)


if __name__ == "__main__":
    main()

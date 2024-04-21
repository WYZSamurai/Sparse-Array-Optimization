import plotly.graph_objects as go
import torch
import pattern
import msll


def fitness(dna: torch.Tensor, lamb: float, d: float, delta: int, theta_0: float):
    phase0 = torch.zeros_like(dna)
    Fdb = pattern.pattern(dna, phase0, lamb, d, delta, theta_0)
    fit = -msll.msll(Fdb)

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
    # 确定每对个体是否进行交换
    swap_flags = torch.rand(NP // 2) < Pc
    # 需要进行交换的个体对的索引
    swap_indices = torch.where(swap_flags)[0] * 2
    # 对于每一对需要进行交换的个体，生成一个随机的交换点
    randcuts = torch.randint(1, ME-1, (swap_flags.sum(),))

    # 执行交换操作
    for idx, randcut in zip(swap_indices, randcuts):
        # 交换基因片段
        temp = dna[idx, randcut:].clone()
        dna[idx, randcut:], dna[idx+1, randcut:] = dna[idx+1, randcut:], temp

    return dna


def mutation(dna: torch.Tensor, Pm: float):
    # (1,ME-2)
    P = (torch.rand(dna.shape, device=dna.device) -
         Pm * torch.ones(dna.shape, device=dna.device))
    dna = torch.where(P > 0, input=dna, other=1-dna)
    dna[:, [0, -1]] = 1
    return dna


def judge(dna: torch.Tensor, NE: int):
    # 无bug
    NP, ME = dna.shape
    ones = torch.ones(ME, dtype=dna.dtype, device=dna.device)

    # 计算每个序列中1的总数,计算每个序列与NE的差异
    difference = torch.matmul(dna, ones)-NE

    for i in range(NP):
        if difference[i] > 0:
            # 需要移除的1的数量
            num_to_remove = int(difference[i].item())
            # 找出[1,ME-2]1的索引
            ones_idx = torch.where(dna[i][1:ME-1] == 1)[0]+1
            # 随机选择一些1转换为0
            if len(ones_idx) > 0:
                remove_idx = ones_idx[torch.randperm(len(ones_idx))[
                    :num_to_remove]]
                dna[i, remove_idx] = 0
        elif difference[i] < 0:
            # 需要添加的1的数量
            num_to_add = int(-difference[i].item())
            # 找出所有0的索引
            zeros_idx = torch.where(dna[i] == 0)[0]
            # 随机选择一些0转换为1
            if len(zeros_idx) > 0:
                add_idx = zeros_idx[torch.randperm(
                    len(zeros_idx))[:num_to_add]]
                dna[i, add_idx] = 1
    return dna


def GA(dna: torch.Tensor, G: int, Pc: float, Pm: float, NE: int, lamb: float, d: float, delta: int, theta_0: float):
    ME = dna.shape[1]
    ybest = torch.zeros(G,)
    dnabest = torch.zeros(G, ME)

    print("开始优化")
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

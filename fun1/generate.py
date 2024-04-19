import torch


def gen(NP: int, ME: int, NE: int):
    dna = torch.zeros(NP, ME)

    # 参数检查
    if NE < 2 or ME < NE:
        print("参数不满足条件，NE至少为2且ME至少为NE。")
        return

    for i in range(NP):
        idx = (torch.randperm(ME-2)+1)[:NE-2]
        dna[i, idx] = 1

    dna[:, -1] = 1
    dna[:, 0] = 1

    return dna

import torch


def gen(NP: int, ME: int, NE: int):
    dna = torch.zeros(NP, ME)

    # 设置首列和尾列为1
    dna[:, [0, -1]] = 1

    # 随机生成中间列的1
    for i in range(NP):
        # 生成一个不重复的随机索引序列
        indices = torch.randperm(ME-2)[:NE-2] + 1  # 加1是为了避开第一列
        # 将选中的位置设置为1
        dna[i, indices] = 1
    dna = dna.reshape(NP, 1, 1, ME)
    return dna


if __name__ == "__main__":
    pass

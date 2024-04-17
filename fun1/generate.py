import torch


def gen(NP: int, ME: int, NE: int):
    # 初始化全零矩阵
    dna = torch.zeros(NP, ME)

    # 设置首列和尾列为1
    dna[:, [0, -1]] = 1

    # 每行需要设置为1的中间元素数量
    inner_ones_count = NE - 2

    # 生成每行的随机索引
    # 注意：这里使用 torch.randint 避免索引重复，每行的索引是独立的
    random_indices = torch.randint(1, ME-1, (NP, inner_ones_count))

    # 构建一个批量索引数组，用于定位每行的索引位置
    batch_indices = torch.arange(NP).unsqueeze(1).expand(-1, inner_ones_count)

    # 使用高级索引一次性设置中间的1
    dna[batch_indices, random_indices] = 1

    return dna

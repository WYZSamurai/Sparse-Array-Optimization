import torch


# 计算最大副瓣电平
def msll(Fdb: torch.Tensor):
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
    mask &= ~((ranges >= (maxi - indexl).unsqueeze(1)) &
              (ranges <= (maxi + indexr).unsqueeze(1)))

    Fdb = Fdb.masked_fill(~mask, float('-inf'))

    # 寻找调整后的最大电平(batch_size,)
    MSLL = Fdb.max(dim=1).values

    return MSLL

import generate


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
    AA = d*(ME-1)

    dna = generate.gen(NP, ME, NE)

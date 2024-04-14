import generate
import GA
import pattern


if __name__ == "__main__":
    # 种群数
    NP = 100
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
    delta = 3600
    # 阵列口径（米）
    # AA = d*(ME-1)
    # 角度范围（角度）
    theta_min = -90.0
    theta_max = 90.0

    # dna(NP,ME)
    dna = generate.gen(NP, ME, NE)

    ybest, dnabest = GA.GA(dna, G, Pc, Pm, NE, lamb, d, delta, theta_0)

    Fdb1 = pattern.pattern(dna[0], lamb, d, delta, theta_0)
    pattern.plot(Fdb1, delta, theta_min, theta_max)

    Fdb2 = pattern.pattern(dnabest, lamb, d, delta, theta_0)
    pattern.plot(Fdb2, delta, theta_min, theta_max)

    GA.plot(G, ybest)
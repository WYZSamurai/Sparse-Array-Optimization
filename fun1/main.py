import generate
import GA
import pattern
import torch


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")


if __name__ == "__main__":
    # 种群数
    NP = 100
    # 交叉率
    Pc = 0.8
    # 变异率
    Pm = 0.050
    # 迭代次数
    G = 20

    # 实际阵元个数
    NE = 24
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
    dna = generate.gen(NP, ME, NE).to(device)
    phase0 = torch.zeros_like(dna)

    ybest, dnabest = GA.GA(dna, G, Pc, Pm, NE, lamb, d, delta, theta_0)
    save_path = "fun1/bestdna.pth"
    print("保存最佳阵元位置。。。。。")
    torch.save(dnabest, save_path)
    print("保存完成")

    print("生成优化前方向图。。。。。")
    Fdb1 = pattern.pattern(dna, phase0, lamb, d, delta, theta_0)
    pattern.plot(Fdb1[0], delta, theta_min, theta_max)
    print("保存完成")

    print("生成优化后方向图。。。。。")
    Fdb2 = pattern.pattern(dnabest.to(device=dna.device).reshape(
        1, ME), phase0, lamb, d, delta, theta_0)
    pattern.plot(Fdb2[0], delta, theta_min, theta_max)
    print("保存完成")

    GA.plot(G, ybest)

import torch
import pattern


def msll(Fdb: torch.Tensor):
    maxi = Fdb.argmax()
    indexr = torch.where(Fdb[maxi:].diff() > 0)[0][0]
    if maxi == torch.tensor(0):
        indexl = maxi
    else:
        indexl = torch.where(Fdb[:maxi].flip(0).diff() > 0)[0][0]
    Fdb[maxi-indexl:maxi+indexr] = -60
    mFdb = Fdb.max()
    MSLL = mFdb
    return MSLL


if __name__ == "__main__":
    theta_min = -90.0
    theta_max = 90.0
    (l, delta, theta_0) = (1, 360, 0)
    d = l/2
    G = 10000
    NP = 100
    m = 1
    n = 1
    L = 20
    Pc = 0.8
    Pm = 0.050

    for i in range(G):
        dna = torch.randint(0, 2, (m, n, L)).to(dtype=torch.float)
        Fdb = pattern.pattern(dna, l, d, delta, theta_0)
        print("第", i+1, "个Fdb为：\n", Fdb)
        MSLL = msll(Fdb)
        print("第", i+1, "个MSLL为：\n", MSLL)

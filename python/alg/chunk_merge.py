# input P
import math

def chunk_merge(P, Ls, Lp, ts, tp, r=0.25, d=0.2):
    """
    Ls=Lp/r, ts=(1+d)tp, P_j = (s_j, e_j, pt_j), pt_j=0/1/2 (p/sr/n)
    output:
        C, G, t[L]
        G: 0/1 (sr/p)
    """
    inf = 100000.
    L = P[-1][1]+1
    m = len(P)
    T = [inf] * L
    dec = [None]*L
    def OPT(i, j):
        # reuse
        if T[i] < inf:
            return T[i]
        # compute
        if j<0 or (j==0 and P[j][2]==0):
            return 0
        # SR_T
        ori_j = j
        q = Ls
        ls = 0
        while q>0:
            if j<0 or (j==0 and P[j][2]==0):
                break
            if P[j][2]==2:
                # new
                ls += min(math.floor(q*r), i-ls-P[j][0]+1)
                q -= math.ceil(ls/r)
                if i-ls < P[j][0]:
                    j-=1
            else:
                # non-prefix
                ls += min(q, i-ls-P[j][0]+1)
                q -= ls
                if i-ls < P[j][0]:
                    j-=1
        sr_j = j
        SR_T = OPT(i-ls, j) + ts
        # P_T
        j = ori_j
        q = Lp
        lp = 0
        while q>0:
            if j<0 or (j==0 and P[j][2]==0):
                break
            # same
            lp += min(q, i-lp-P[j][0]+1)
            q -= lp
            if i-lp < P[j][0]:
                j-=1
        p_j = j
        P_T = OPT(i-lp, j) + tp
        if SR_T < P_T:
            dec[i] = (i-ls, 0)
            T[i] = SR_T
        else:
            dec[i] = (i-lp, 1)
            T[i] = P_T
        return T[i]
    def trace():
        i = L-1
        C,G = [], []
        while i>=0 and dec[i] is not None:
            c, g = dec[i]
            C.insert(0, (c+1, i))
            G.insert(0, g)
            print(i, c, g)
            i = c
        return C, G
    OPT(L-1, m-1)
    C, G = trace()
    return C, G, T[L-1]

if __name__ == "__main__":
    P = [(0, 5, 0), (6, 11, 1), (12, 15, 2), (16, 19, 1), (20, 22, 2)]
    Ls = 4
    Lp = 2
    ts = 0.12
    tp = 0.1
    r = 0.5
    d = 0.2
    C,G,minT = chunk_merge(P, Ls, Lp, ts, tp, r, d)
    print("C:", C)
    print("G:", G)
    print("minT:", minT)
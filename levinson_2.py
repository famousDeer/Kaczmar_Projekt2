import numpy as np

def levinson_2(seg, N, ar_rank):
    p = np.zeros(ar_rank+1)
    for i in range(ar_rank+1):
        for t in range(i, N):
            p[i] = p[i] + seg[t] * seg[t-i]
    a = np.zeros((ar_rank, ar_rank))
    s = np.zeros(11)
    k = np.zeros(10)
    s[0] = p[0]
    k[0] = p[1]/p[0]
    a[0,0] = k[0]
    s[1] = (1-k[0]**2)*s[0]
    for i in range(1, ar_rank):
        x = 0
        for j in range(0, i):
            x = x + a[j,i-1]*p[i-j]
        k[i] = (p[i+1]-x)/s[i]
        a[i,i] = k[i]
        for j in range(0, i):
            a[j,i] = a[j, i-1] - k[i]*a[i-j-1, i-1]
        s[i+1] = (1-k[i]**2) * s[i]
    return k


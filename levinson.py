import numpy as np

def levinson(seg: list, samples_ammount: int, ar_rank: int):
    """Levinson-Durbin algorithm

    Args:
        seg (list): Data Segment
        samples_ammount (int): Number of samples in segment
        ar_rank (int): AR rank

    Returns:
        k (list): Reflection coeeficients
    """
    # Calculate autocorrelation coefficients
    p = np.zeros(ar_rank+1)
    for i in range(ar_rank+1):
        for t in range(i, samples_ammount):
            p[i] = p[i] + seg[t] * seg[t-i]
    
    # Solving Yole-Walker equations
    a = np.zeros((ar_rank, ar_rank))
    sigma = np.zeros(11)
    k = np.zeros(10)
    sigma[0] = p[0]
    k[0] = p[1]/p[0]
    a[0,0] = k[0]
    sigma[1] = (1-k[0]**2)*sigma[0]
    for i in range(1, ar_rank):
        x = 0
        for j in range(0, i):
            x = x + a[j,i-1]*p[i-j]
        k[i] = (p[i+1]-x)/sigma[i]
        a[i,i] = k[i]
        for j in range(0, i):
            a[j,i] = a[j, i-1] - k[i]*a[i-j-1, i-1]
        sigma[i+1] = (1-k[i]**2) * sigma[i]
    
    return k


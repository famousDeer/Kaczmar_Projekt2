import numpy as np
from Levinson_Durbin.utils import acovf

def levinson_durbin_alg(s, order=10):

    # order = nlags

    sxx_m = acovf(s, fft=False)[: order + 1]

    phi = np.zeros((order + 1, order + 1), "d")
    sig = np.zeros(order + 1)

    phi[1, 1] = sxx_m[1] / sxx_m[0]
    sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
    for k in range(2, order + 1):
        phi[k, k] = (
            sxx_m[k] - np.dot(phi[1:k, k - 1], sxx_m[1:k][::-1])
        ) / sig[k - 1]
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)

    arcoefs = phi[1:, -1]

    return arcoefs
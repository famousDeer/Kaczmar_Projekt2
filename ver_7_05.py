import numpy as np
from scipy.io.wavfile import read, write
from levinson_2 import levinson_2
from kodowanie import koduj, dekoduj, czytaj_plik, zapisz_plik
import matplotlib.pyplot as plt


def nad(samples_ammount, ar_rank):
    track = read("/data/01.wav")
    input_arr = np.array(track[1], dtype=float)
    input_arr = 0.5 * input_arr / 32768
    input_arr = np.array(input_arr)

    t = np.arange(samples_ammount)
    k = np.arange(samples_ammount)

    segment_id = 0
    weight_function = 0.5 * (1 - np.cos(2*np.pi*k/(samples_ammount+1)))
    weight_function = np.roll(weight_function, -1)
    e_g = np.zeros((len(input_arr) // samples_ammount) * samples_ammount + 1)
    e_max_g = np.zeros(len(input_arr) // samples_ammount)
    a_g = np.zeros((len(input_arr) // samples_ammount + 1) * ar_rank)
    print(len(e_g), len(e_max_g), len(a_g))

    while t[-1] < len(input_arr):
        segment_id += 1

        y = input_arr[t]
        yw = y * weight_function
        yr = np.concatenate((np.zeros(ar_rank), yw, np.zeros(ar_rank)), axis=0)

        e = np.zeros(samples_ammount)

        track = np.zeros(ar_rank)
        r2 =  np.correlate(y, y, mode='full')
        lg = np.arange(-255, 256)
        r2 = r2[lg >= 0]
        # a = ld(r2, 10)[1:]
        track = levinson_2(yr, samples_ammount, 10)
        if segment_id == 1:
            y = np.concatenate((np.zeros(ar_rank), y), axis=0)
        else:
            y = np.concatenate((input_arr[t[0]-ar_rank:t[0]], y), axis=0)

        for i in range(samples_ammount):
            e[i] = y[i+ar_rank] + np.sum(y[i:i+ar_rank][::-1] * track)
        e_g[t] = e

        e_max = max(abs(e))
        e_max_g[segment_id-1] = e_max
        a_g[(segment_id-1)*ar_rank:segment_id*ar_rank] = track

        t = t + samples_ammount

    return input_arr, a_g, e_max_g, e_g


def odb(a, e_max, e):
    samples_ammount = 256
    r = 10
    segment_n = len(e_max)
    y_o = np.zeros(r + segment_n*samples_ammount + 1)

    for i in range(segment_n):
        p_s = r + i * samples_ammount
        y_s = np.zeros(samples_ammount)
        a_s = a[i*r:(i+1)*r]
        for j in range(samples_ammount):
            p_s2 = p_s + j
            # p1 = y_o[p_s2-r:p_s2][::-1]
            # p2 = e[i * samples_ammount + j]
            y_s[j] = -np.sum(y_o[p_s2-r:p_s2][::-1] * a_s) + e[i * samples_ammount + j]
        y_o[p_s:p_s+samples_ammount] = y_s

    write('wiersz_reconstructed_1b.wav', rate=11025, data=y_o)

    return y_o


if __name__ == '__main__':
    bits = 8
    samples_ammount = 256
    ar_rank = 10
    inp, a, e_max, e = nad(samples_ammount, ar_rank)

    # zapisz e_max
    bits_e_max=16
    N_e_max = len(e_max)
    err_max_s = koduj(e_max, lb=bits_e_max, e_max=max(abs(e_max)))
    zapisz_plik(err_max_s, 'e_max_'+str(bits)+'.bin')

    # zapisz współczynniki
    bits_coefs=16
    N_coefs = len(a)
    a_s = koduj(a, lb=bits_e_max, e_max=max(abs(a)))
    zapisz_plik(a_s, 'coefs_'+str(bits)+'.bin')

    # zapisz błędy
    err_s = ""
    for i in range(len(e_max)):
        err_s += koduj(e[i*samples_ammount:(i+1)*samples_ammount], lb=bits, e_max=e_max[i])
    zapisz_plik(err_s, 'errors_'+str(bits)+'.bin')

    # czytaj e_max
    err_max_sr = czytaj_plik('e_max_'+str(bits)+'.bin')
    errors_max = dekoduj(err_max_sr[0:N_e_max*bits_e_max+16], lb=bits_e_max, l=N_e_max, e_max=max(abs(e_max)))

    # czytaj wzpółczynniki
    a_sr = czytaj_plik('coefs_'+str(bits)+'.bin')
    a_coefs = dekoduj(a_sr[0:N_coefs*bits_coefs+16], lb=bits_coefs, l=N_coefs, e_max=max(abs(a)))

    # czytaj błędy
    err_sr = czytaj_plik('errors_'+str(bits)+'.bin')
    errors = np.array([])
    add = 0
    for i in range(len(errors_max)):
        errors = np.append(errors, dekoduj(err_sr[i*samples_ammount*bits+add:(i+1)*samples_ammount*bits+add+16], lb=bits, l=samples_ammount, e_max=errors_max[i]))
        add += 16



    out = odb(a_coefs, errors_max, errors)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(inp[0:550])
    ax.plot(out[10:560])
    plt.xlabel('Numer próbki')
    plt.savefig(str(bits)+'_bit.png')
    plt.show()


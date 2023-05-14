import numpy as np

def zapisz_plik(s, name = 'file_test_2.bin'):
    i = 0
    buffer = bytearray()
    while i < len(s):
        buffer.append(int(s[i:i + 8], 2))
        i += 8

    with open(name, 'bw') as f:
        f.write(buffer)

def czytaj_plik(name = 'file_test_2.bin'):
    with open(name, 'rb') as f:
        fff = f.read()
        return (bin(int.from_bytes(fff, byteorder='big')).replace('0b', ''))


def koduj(arr, lb, e_max):
    arr = arr / e_max
    l_n = 2**lb
    levels = np.linspace(-1, 1, l_n)
    str = "1"
    for item in arr:
        val = np.abs(levels - item).argmin()
        str += f'0b{val:16b}'[2:].replace(' ', '0')[-lb:]
    while len(str) % 16 != 0:
        str = str + '0'
    return str

def dekoduj(str, lb, l, e_max):
    l_n = 2 ** lb
    levels = np.linspace(-1, 1, l_n)
    zeros = ""
    result = []
    for i in range(2-lb):
        zeros += "0"
    for i in range(l):
        s = zeros + str[i*lb+1:(i+1)*lb+1]
        # print(s)
        num = int(s, 2)
        result.append(levels[num])
    result = np.array(result) * e_max
    return result
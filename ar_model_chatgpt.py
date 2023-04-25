import numpy as np
import scipy.io.wavfile as wavfile
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from Levinson_Durbin import LD_alg
from time import time
from logging import getLogger

logger = getLogger('dummy-logger')

# Load the audio signal
start_time = time()
logger.warning(f'Reading wave file') 
fs, signal = wavfile.read('data/new.wav')
read_time = time()
reading_completion_time = read_time - start_time
logger.warning(f'Wave file read succesfuly in {reading_completion_time} seconds')

# Choose the order of the AR model
p = 10

# Compute the autocorrelation coefficients
second_time = time()
ar_coeff = np.correlate(signal, signal, mode='full')
ar_coeff = ar_coeff[len(ar_coeff)//2:]
computed_coeff = time()
complete_computing_coeff = computed_coeff - second_time
logger.warning(f'Autocorrelation coefficients computed succesfuly in {complete_computing_coeff} seconds')
logger.warning(f'Autocorrelation coefficients {ar_coeff}')

# Compute the autocorrelation matrix
ar_matrix = linalg.toeplitz(ar_coeff[:p])
logger.warning(f'Autocorrelation matrix {ar_matrix}')

# Solve the Yule-Walker equations
yw_equations = np.linalg.solve(ar_matrix, ar_coeff[1:p+1])
logger.warning(f"Yule-Walker equations {yw_equations}")

# Generate the AR signal
ar_signal = np.zeros_like(signal)
for i in range(p, len(signal)):
    ar_signal[i] = np.dot(yw_equations, signal[i-p:i])

# Compute the residual signal
residual = signal - ar_signal

# Optional: Compute the spectrum of the AR and residual signals
ar_spectrum = np.abs(np.fft.fft(ar_signal))**2
residual_spectrum = np.abs(np.fft.fft(residual))**2

ld_solver = LD_alg.LevinsonDurbinRecursion(ar_matrix)
ld_solver.toeplitz_elements = ld_solver.toeplitz_elements[0]
result = ld_solver.solve()
logger.warning(f"Levinson-Durbin results = {result}")

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from math import cos, pi
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import levinson_durbin
from Levinson_Durbin import levinson 
import IPython


#  ========== FUNCTIONS ========== 
# Weight for flattening edges
def weight(i: int):
    return 0.5*(1-cos(((2*pi) / (257)) * i))

# Check if last 10 smaples from previous segment
# fir first 10 samples from current segment
def check_samples(segments: np.ndarray):
    for i in range(1,len(segments)):
        if list(segments[i-1][-10:]) != list(segments[i][:10]):
            print('\033[91m' + "X Test 1 Failed")
            print(f"List #{i-1} is diff with list #{i}")
    print('\033[92m' + f"\u2713 Test Passed")

def play_sound(sound, rate=11025):
    return  IPython.display.display(IPython.display.Audio(sound, rate=rate))
# Read wave file
track, fs = sf.read("data/01.wav")

play_sound(track[:256])

# Ploting track with specgram
plt.figure(figsize=(12,6))

plt.subplot(2, 1, 1)
plt.plot(track)

plt.subplot(2, 1, 2)
plt.specgram(track, NFFT=1024, Fs=fs)

# plt.show()

# Creating segments with 256 samples 
segments_clear = [track[:256]]
for i in range(256, len(track), 256):
    segments_clear.append(track[i-10 : i+256])

# Adding 0 to fit 256
last_seg_len = (len(segments_clear[-1]) - 256 ) * -1
if last_seg_len > 0:
    segments_clear[-1] = np.append(segments_clear[-1], [0]*last_seg_len)

# Check if last 10 samples from previus segment
# fit first 10 samples from current segment
check_samples(segments_clear)

# Create segments model with zeros 
segments_model = []

for idx, lst in enumerate(segments_clear):
    lst[0] = weight(1)
    lst[-1] = weight(256)
    segments_model.append(np.zeros(10))
    segments_model[idx] = np.append(segments_model[idx], lst)
    segments_model[idx] = np.append(segments_model[idx], np.zeros(10))


# Check if zeros added to segments
check_samples(segments_model)

sigma_v, arcoefs, pacf, sigma, phi = levinson_durbin(segments_model[0],10)

print(arcoefs)
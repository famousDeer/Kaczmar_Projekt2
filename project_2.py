import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Read wave file
track, fs = sf.read("data/01.wav")

# Ploting track with specgram
plt.figure(figsize=(12,6))

plt.subplot(2, 1, 1)
plt.plot(track)

plt.subplot(2, 1, 2)
plt.specgram(track, NFFT=1024, Fs=fs)

plt.show()

# Creating segments with 256 samples 
segments = [track[:256]]
for i in range(256, len(track), 256):
    segments.append(track[i-10 : i+256])

# Adding 0 to fit 256
last_seg_len = (len(segments[-1]) - 256 ) * -1
if last_seg_len > 0:
    segments[-1] = np.append(segments[-1], [0]*last_seg_len)

# Check if last 10 samples from previus segment
# fit first 10 samples from current segment
for i in range(1,len(segments)):
    if any(segments[i-1]) != any(segments[i]):
        print('\033[91m' + "X Test 1 Failed")
        print(f"List #{i-1} is diff with list #{i}")
print('\033[92m' + "\u2713 Test 1 Passed")


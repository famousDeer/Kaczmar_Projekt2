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
segments_clear = [track[:256]]
for i in range(256, len(track), 256):
    segments_clear.append(track[i-10 : i+256])

# Adding 0 to fit 256
last_seg_len = (len(segments_clear[-1]) - 256 ) * -1
if last_seg_len > 0:
    segments_clear[-1] = np.append(segments_clear[-1], [0]*last_seg_len)

# Check if last 10 samples from previus segment
# fit first 10 samples from current segment
for i in range(1,len(segments_clear)):
    if any(segments_clear[i-1]) != any(segments_clear[i]):
        print('\033[91m' + "X Test 1 Failed")
        print(f"List #{i-1} is diff with list #{i}")
print('\033[92m' + "\u2713 Test 1 Passed")

# Create segments model with zeros 
segments_model = []

for idx, lst in enumerate(segments_clear):
    segments_model.append(np.zeros(10))
    segments_model[idx] = np.append(segments_model[idx], lst)
    segments_model[idx] = np.append(segments_model[idx], np.zeros(10))

# Check if zeros added to segments
for i in range(1,len(segments_model)):
    if any(segments_model[i-1]) != any(segments_model[i]):
        print('\033[91m' + "X Test 1 Failed")
        print(f"List #{i-1} is diff with list #{i}")
print('\033[92m' + "\u2713 Test 2 Passed")

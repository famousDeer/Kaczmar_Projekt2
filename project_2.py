import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from math import cos, pi
from sklearn.linear_model import LinearRegression


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

class AR:
  def __init__(self, order):
    self.order = order
    self.model = LinearRegression()
    self.sigma = None

  def generate_train_x(self, X):
    n = len(X)
    ans = X[:n-self.order]
    ans = np.reshape(ans, (-1, 1))
    for k in range(1, self.order):
      temp = X[k:n-self.order+k]
      temp = np.reshape(temp, (-1, 1))
      ans = np.hstack((ans, temp))
    return ans
  
  def generate_train_y(self, X):
    return X[self.order:]

  def fit(self, X):
    self.sigma = np.std(X)
    train_x = self.generate_train_x(X)
    train_y = self.generate_train_y(X)
    self.model.fit(train_x, train_y)

  def predict(self, X, num_predictions, mc_depth):
    X = np.array(X)
    ans = np.array([])

    for j in range(mc_depth):
      ans_temp = []
      a = X[-self.order:]

      for i in range(num_predictions):
        next = self.model.predict(np.reshape(a, (1, -1))) + np.random.normal(loc=0, scale=self.sigma)

        ans_temp.append(next)
        
        a = np.roll(a, -1)
        a[-1] = next
      
      if j==0:
        ans = np.array(ans_temp)
      
      else:
        ans += np.array(ans_temp)
    
    ans /= mc_depth

    return ans

# Read wave file
track, fs = sf.read("data/01.wav")

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

model = AR(10)
model.fit(segments_model[0])
prediction = model.predict(segments_model[0][:10], 100, 1)
prediction = np.reshape(prediction, (-1,))
prediction = np.hstack((segments_model[0][:10], prediction))

plt.figure(figsize=(24, 16))
plt.plot(prediction)
plt.show()
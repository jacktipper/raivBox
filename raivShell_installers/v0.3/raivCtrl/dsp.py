import numpy as np
import soundfile as sf
import librosa
import warnings
import os
warnings.filterwarnings('ignore')

try:
    os.remove('Audio/output.wav')
except:
    pass

x, fs = librosa.load('Audio/input.wav',
                     sr=None, mono=True)

threshold = 0.15
norm = 0.8
N = int(len(x))

for n in range(N):
    if np.abs(x[n] > threshold):
        x[n] = x[n] / np.abs(x[n]) * threshold
    x[n] = x[n] / threshold * norm

sf.write('Audio/output.wav', x, fs)

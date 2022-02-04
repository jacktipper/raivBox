import Jetson.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
led0pin = 20
GPIO.setup(led0pin, GPIO.OUT, initial=GPIO.LOW)

import numpy as np
import soundfile as sf
import librosa
import warnings
import os
os.system('./init-py.sh')
warnings.filterwarnings('ignore')

from datetime import datetime

try:
    dest = str(datetime.now())[0:19]
    os.rename('audio/rendered/output.wav', str('audio/archive/' + dest + '.wav'))
except:
    pass

x, fs = librosa.load('Audio/input.wav',
                     sr=None, mono=True)

threshold = 0.05
norm = 0.8
N = int(len(x))

for n in range(N):
    if np.abs(x[n] > threshold):
        x[n] = x[n] / np.abs(x[n]) * threshold
    x[n] = x[n] / threshold * norm

sf.write('Audio/output.wav', x, fs)

GPIO.output(led0pin, GPIO.HIGH)

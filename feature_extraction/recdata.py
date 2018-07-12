import os, sys, math, select
import numpy as np
import librosa
import sounddevice as sd
import time as tm

if not os.path.exists('temp_files'):
    os.makedirs('temp_files')

def record(time = 1, fs = 44100):
    #os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'feature_extraction'))
    file = 'temp_out'
    duration = time
    recording = sd.rec(int(duration*fs),samplerate=fs, channels=1, blocking  = False)
    for i in range(time):
        i += 1
        tm.sleep(1)
    recording = recording[:,0]
    np.save(file,recording)
    np.seterr(divide='ignore', invalid='ignore')
    scaled = np.int16(recording/np.max(np.abs(recordig)) * 32767)
    wavf.write(file+'.wav', fs, scaled)
    #root, dirs, files = os.walk("").next()
    #path = os.path.join(root,file)
    data, fs = librosa.load(file+'.wav')
    os.remove(file+'.npy')
    os.remove(file+'.wav')
    return data, fs

def readaudio(file):
    data, fs = librosa.load(file)
    os.remove(file)
    return data, fs


import os, sys, math, select
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import time as tm

sys.path.append("ssloc")

#from ssloc.soundrecorder1 import recorder
from sklearn import svm
from sklearn.externals import joblib
import pickle

clf = joblib.load('input/detection.pkl')

def record(time=10, fs=44100):
    duration = time
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking = False)
    for i in range(time):
        #sys.stdout.write('\r%d seconds elapsed' % (i+1))
        #sys.stdout.flush()
        i += 1
        tm.sleep(1)

    recording = recording[:, 0]
    #np.save(filename, recording)
    return recording

def extract_features(rec_data, s_r):
    #getting features
    st_ft = np.abs(librosa.stft(rec_data))
    mfccs = np.mean(librosa.feature.mfcc(y = rec_data, sr = s_r, n_mfcc = 40).T, axis = 0)
    chroma = np.mean(librosa.feature.chroma_stft(S = st_ft, sr = s_r).T, axis = 0)
    mel = np.mean(librosa.feature.melspectrogram(rec_data, sr= s_r).T, axis = 0)
    contrast = np.mean(librosa.feature.spectral_contrast(S = st_ft, sr= s_r).T, axis = 0)
    tonnetz = np.mean(librosa.feature.tonnetz(y = librosa.effects.harmonic(rec_data), sr= s_r).T, axis = 0)
    mfcc_data = []
    features = np.empty((0,193))
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = np.vstack([features, ext_features])
    cols = ["features", "shape"]
    mfcc_data.append([features, features.shape])
    mfcc_pd = pd.DataFrame(data = mfcc_data, columns = cols)
    flat = [mfcc_pd['features'][i].ravel() for i in range(mfcc_pd.shape[0])]
    mfcc_pd['sample'] = pd.Series(flat, index = mfcc_pd.index)
    test_data = np.array(list(mfcc_pd[:]['sample']))
    return test_data

def prediction_label(value):
    if value == 0:
        label = "far"
    elif value == 1:
        label = "midrange"
    elif value == 2:
        label = "near"
    elif value == 3:
        label = "vfar"
    elif value == 4:
        label = "vnear"
    return label


#sr = recorder()
#sr.setup()
#raw_data = sr.getAudio()
#s_r = sr.RATE
#raw_data = raw_data.astype(float)
#sr.close()
#test = extract_features(raw_data, s_r)
#x = clf.predict(test)
    
i = 0
while True:
    #os.system('cls' if os.name == 'nt' else 'clear')
    record1 = record()
    test1 = extract_features(record1, 44100)
    x1 = clf.predict(test1)
    sys.stdout.write('\r The drone is %s \r \r'% prediction_label(x1[0]))
    sys.stdout.flush()
    record1 = []
    i += 1
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = input()
        break

print(i)
#print(prediction_label(x))


    





import os, sys, math
import numpy as np
import pandas as pd
import librosa

sys.path.append("ssloc")

from ssloc.sounderecorder1 import recorder
from sklearn import svm
from sklearn.externals import joblib
import pickle

clf = joblib.load('input/detection.pkl')

def extract_features(rec_data, s_r):
    #getting features
    st_ft = np.abs(librosa.stft(rec_data))
    mfccs = np.mean(librosa.feature.mfcc(y = rec_data, sr = s_r, n_mfcc = 40).T, axis = 0)
    chroma = np.mean(librosa.feature.chrome_stft(S = st_ft, sr = s_r).T, axis = 0)
    mel = np.mean(librosa.feature.melspectogram(rec_data, sr= s_r).T, axis = 0)
    contrast = np.mean(librosa.feature.spectral_contrast(S = st_ft, sr= s_r).T, axis = 0)
    tonnetz = np.mean(librosa.feature.tonnetz(Y = librosa.effects.harmonic(rec_data, sr= s_r).T, axis = 0)

    #manipulate features for testing based on trained pickled data available(refer training.py)
    mfcc_data = []
    features =np.empty((0,193))
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = np.vstack([features, ext_features])
    cols = ["features", "shape"]
    mfcc_data.append([features, features.shape])
    mfcc_pd = pd.DataFrame(data = mfcc_data, columns = cols)
    flat = [mfcc_pd['features'][i].ravel() for i in range(mfcc_pd.shape[0])]
    mfcc_pd['sample'] = pd.Series(flat, undex = mfcc_pd.index)
    test_data = np.array(list(mfcc_pd[:]['sample']))
    return test_data




    





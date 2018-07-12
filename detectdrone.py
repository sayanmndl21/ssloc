import os, sys, math, select
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import time as tm
import scipy.io.wavfile as wavf

from feature_extraction import filters as fil
from feature_extraction import lpcgen as lpg
from feature_extraction import calcspectrum as csp
from feature_extraction import harmonics as hmn
from feature_extraction import fextract as fex
from feature_extraction import parsedata as par

from sklearn import svm
from sklearn.externals import joblib
import pickle

clf = joblib.load('input/detection_iris_new.pkl')
clf1 = joblib.load('input/detect_irisdrone_new.pkl')

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
    scaled = np.int16(recording/np.max(np.abs(recording)) * 32767)
    wavf.write(file+'.wav', fs, scaled)
    #root, dirs, files = os.walk("").next()
    #path = os.path.join(root,file)
    data, fs = librosa.load(file+'.wav')
    os.remove(file+'.npy')
    os.remove(file+'.wav')
    return data, fs

def dist_prediction_label(value):
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

def drone_prediction_label(value):
    if value == 0:
        label = "drone"
    elif value == 1:
        label = "no drone"
    return label

i = 0
bandpass = [600,10000]
while True:
    data, fs = record()
    ns = fil.bandpass_filter(data,bandpass)
    try:
        p,b = hmn.psddetectionresults(data)
    except IndexError:
        pass
        b = False

    
    if b:
        mfcc, chroma, mel, spect, tonnetz = fex.extract_feature(ns,fs)
        a,e,k = lpg.lpc(ns,10)
        mfcc_test = par.get_parsed_mfccdata(mfcc, chroma,mel,spect,tonnetz)
        lpc_test = par.get_parsed_lpcdata(a,k,p)
        sys.stdout.write("\r Maybe a drone... Please Wait \r \r \r \n \r")
        x1 = clf.predict(mfcc_test)
        x2 = clf1.predict(lpc_test)
        sys.stdout.write('\r The drone is %s \r \r \n \r'% dist_prediction_label(x1[0]))
        sys.stdout.write('\r To be sure there is a %s \r \r \n \r'% drone_prediction_label(x2[0]))
        sys.stdout.flush()
    else:
        sys.stdout.write("\ n \r Need time to compute, but I think there is no drone \r \r \r \n")
        sys.stdout.flush()
    i+=1
    if sys.stdin in select.select([sys.stdin],[],[],0)[0]:
        line = input()
        break
print('iter_num:',i)
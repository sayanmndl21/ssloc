import os, sys, math, select
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import time as tm
import scipy.io.wavfile as wavf
import datetime
import warnings

from feature_extraction import filters as fil
from feature_extraction import lpcgen as lpg
from feature_extraction import calcspectrum as csp
from feature_extraction import harmonics as hmn
from feature_extraction import fextract as fex
from feature_extraction import parsedata as par
from feature_extraction.getconfi import logdata
from feature_extraction.apicall import apicalls
from feature_extraction.specsub import reduce_noise

from sklearn import svm
from sklearn.externals import joblib
import pickle

warnings.filterwarnings("ignore")
"""clf = joblib.load('input/detection_iris_new.pkl')## this is the vnear robust one"""
clf = joblib.load('input/detection_backyaardwithnoise.pkl')## this is taken at the beach
#clm = joblib.load('input/detection_new18july.pkl')
#clf1 = joblib.load('input/dronedetectionfinal_new.pkl')

rows = 10
cols = 60
winlist = []
log = logdata(4)

######################################################################################################
global itervalue
itervalue = 0

def record(time = 1, fs = 44100):
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
    """uncomment the following if you want to save data"""
    ##################################################################################################
    #wavf.write(file+'_'+str(itervalue)+'.wav', fs, scaled)
    #data, fs = librosa.load(file+'_'+str(itervalue)+'.wav')
    ######################################################################################################
    data, fs = librosa.load(file+'.wav')
    os.remove(file+'.npy')
    #os.remove(file+'.wav')
    return data, fs

def dist_prediction_label(value):
    if value == 0:
        label = "far"
    elif value == 1:
        label = "midrange"
    elif value == 2:
        label = "near"
    #elif value == 3:
        #label = "vfar or nodrone"
    elif value == 3:
        label = "vnear"
    return label

noise, sfr = record()

# def drone_prediction_label(value):
#     if value == 1:
#         label = "drone"
#     elif value == 0:
#         label = "no drone"
#     return label

######################################################################################################################
"""set api and initiate calls"""
api_url = 'http://mlc67-cmp-00.egr.duke.edu/api/events'
apikey = None
push_url = "https://onesignal.com/api/v1/notifications"
pushkey = None
send = apicalls(api_url,apikey, push_url,pushkey)
log.insertdf(3,str(datetime.datetime.now())[:-7]) #inserted dummy value to eliminate inconsistency
i = 0
bandpass = [800,8500]#filter unwanted frequencies
prev_time= tm.time()#initiate time

"""main code"""
try:#don't want user warnings
    while True:
        data, fs = record()
        out = reduce_noise(data,noise)
        ns = fil.bandpass_filter(data,bandpass)
        try:
            p,freq, b = hmn.psddetectionresults(data)
        except IndexError:
            pass
            b = False
        b = True
        
        if b:
            mfcc, chroma, mel, spect, tonnetz = fex.extract_feature(ns,fs)
            #a,e,k = lpg.lpc(ns,10)
            mfcc_test = par.get_parsed_mfccdata(mfcc, chroma,mel,spect,tonnetz)
            #lpc_test = par.get_parsed_lpcdata(a,k,freq)
            x1 = clf.predict(mfcc_test)
            #x02 = clm.predict(mfcc_test)
            #x1 = ((x01[0]+x01[0])/2)
            #x2 = clf1.predict(lpc_test) 
            print("Drone at %s"% dist_prediction_label(int(x1)))
            log.insertdf(int(x1),str(datetime.datetime.now())[:-7])
            print(x1)
            output = log.get_result()
            '''-----------uncomment if you want to save logs-----------------'''
            #log.logdf(sys.argv[1],x01[0],x02[0],str(datetime.datetime.now())[:-7])
            '''---------------------------------------------------------------'''
            if True:#i > 9:
                print(int(output['Label']))
                #win.addstr(7,5,"Recieved a Result!")
                dt = tm.time() - prev_time
                if dt > 30:#send output every 30secs
                    print('sent %s'% int(output['Label']))
                    #send.sendtoken2(output)
                    prev_time = tm.time()
                    if int(output['Label']) == int(4) or int(output['Label']) == int(2):
                        #send.push_notify()
                        print("pushed %s"% int(output['Label']))
                    #win.addstr(8,5,"Data Sent!")
            ######################################################################################################
            # if itervalue > int(sys.argv[3]):
            #     log.savedf(sys.argv[2])
            #     exit()
            # itervalue+=1
        else:
            print("Wait for result")
        
        
        i+=1


except KeyboardInterrupt:
    pass


print('iter_num:',i)


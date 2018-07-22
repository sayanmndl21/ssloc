import os, sys, math, select
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import time as tm
import scipy.io.wavfile as wavf
import curses, datetime

sys.path.append('labeleddata')

from feature_extraction import filters as fil
from feature_extraction import lpcgen as lpg
from feature_extraction import calcspectrum as csp
from feature_extraction import harmonics as hmn
from feature_extraction import fextract as fex
from feature_extraction import parsedata as par
from feature_extraction import recdata as rdt
from feature_extraction.getconfi import logdata
from feature_extraction.apicall import apicalls

from sklearn import svm
from sklearn.externals import joblib
import pickle

clf = joblib.load('input/detection_iris_new.pkl')
clm = joblib.load('input/detection_new18july.pkl')
clf1 = joblib.load('input/dronedetectionfinal_new.pkl')

rows = 10
cols = 60
winlist = []

global itervalue
itervalue = 0

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
    if value == 1:
        label = "drone"
    elif value == 0:
        label = "no drone"
    return label
log = logdata(10)
log.insertdf(3,str(datetime.datetime.now())[:-7]) #dummy value
i = 0
bandpass = [600,10000]
for root,dirs,files in os.walk(sys.argv[1]):
    for file in files:
        path = os.path.join(root,file)
        try:
            data,fs = rdt.readaudio(path)
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
                #win.addstr(3,5,"Maybe a drone... Please Wait")
                x01 = clf.predict(mfcc_test)
                x02 = clm.predict(mfcc_test)
                x1 = ((x02[0]+x02[0])/2)
                #print("Drone at %s"% dist_prediction_label(int(x1)))
                #soutput = log.get_result()
                '''-----------uncomment if you want to save logs-----------------'''
                log.logdf(sys.argv[2],x01[0],x02[0],str(datetime.datetime.now())[:-7])
                '''---------------------------------------------------------------'''
                #if i > 9:
                #    print(int(output['Label']))
                    #win.addstr(7,5,"Recieved a Result!")
                #    send.sendtoken(output)
                #    if int(output['Label']) != int(3) and int(output['Label']) != int(0):
                #        send.push_notify()
                #        print("sent %s"% int(output['Label']))
                        #win.addstr(8,5,"Data Sent!")
                ######################################################################################################
                #if itervalue > int(sys.argv[3]):
                #    exit()
                #itervalue+=1
            else:
                #win.addstr(3,5,"Maybe a drone... Please Wait")
                #win.addstr(5,5,"Need time to compute, but I think there is no drone")
                #sys.stdout.write("\ n \r Need time to compute, but I think there is no drone \r \r \r \n")
                #sys.stdout.flush()
                print("Wait for result")
        except Exception:
            pass

log.savedf(sys.argv[1])
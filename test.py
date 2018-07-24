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

from sklearn import svm
from sklearn.externals import joblib
import pickle

#curses.initscr()


warnings.filterwarnings("ignore")
clf = joblib.load('input/detection_iris_new.pkl')## this is the vnear robust one
clm = joblib.load('input/detection_new18july.pkl')
#clf1 = joblib.load('input/dronedetectionfinal_new.pkl')

rows = 10
cols = 60
winlist = []
log = logdata(10)

#win = curses.newwin(rows,cols, 10, 3)
#win.clear()
#win.border()
#winlist.append(win)
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
    elif value == 3:
        label = "vfar"
    elif value == 4:
        label = "vnear"
    return label


# def drone_prediction_label(value):
#     if value == 1:
#         label = "drone"
#     elif value == 0:
#         label = "no drone"
#     return label

"""remove mkdir once done"""
#os.mkdir(dist_prediction_label(sys.argv[1]))
###############################################################################################


api_url = 'http://mlc67-cmp-00.egr.duke.edu/api/events'
apikey = None
push_url = "https://onesignal.com/api/v1/notifications"
pushkey = None
send = apicalls(api_url,apikey, push_url,pushkey)
log.insertdf(3,str(datetime.datetime.now())[:-7]) #dummy value
i = 0
bandpass = [600,10000]
prev_time= tm.time()
try:
    while True:
        data, fs = record()
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
            x1 = ((x01[0]+x01[0])/2)
            #x2 = clf1.predict(lpc_test) 
            #win.addstr(5,5,"The drone is %s"% dist_prediction_label(x1[0]))
            #win.addstr(6,5,"To be sure there is a %s "% drone_prediction_label(x2[0]))
            #sys.stdout.write("\r Maybe a drone... Please Wait \r \r \r \n \r")
            #sys.stdout.write('\r The drone is %s \r \r \n \r'% dist_prediction_label(x1[0]))
            #sys.stdout.write('\r To be sure there is a %s \r \r \n \r'% drone_prediction_label(x2[0]))
            #sys.stdout.flush()
            print("Drone at %s"% dist_prediction_label(int(x1)))
            log.insertdf(x1,str(datetime.datetime.now())[:-7])
            print(x1)
            output = log.get_result()
            '''-----------uncomment if you want to save logs-----------------'''
            #log.logdf(sys.argv[1],x01[0],x02[0],str(datetime.datetime.now())[:-7])
            '''---------------------------------------------------------------'''
            if i > 9:
                print(int(output['Label']))
                #win.addstr(7,5,"Recieved a Result!")
                dt = tm.time() - prev_time
                if dt > 30:#send output every 30secs
                    #print('10sec elapsed')
                    send.sendtoken(output)
                    prev_time = tm.time()
                if int(output['Label']) != int(3) and int(output['Label']) != int(0):
                    #send.push_notify()
                    print("sent %s"% int(output['Label']))
                    #win.addstr(8,5,"Data Sent!")
            ######################################################################################################
            # if itervalue > int(sys.argv[3]):
            #     log.savedf(sys.argv[2])
            #     exit()
            # itervalue+=1
        else:
            #win.addstr(3,5,"Maybe a drone... Please Wait")
            #win.addstr(5,5,"Need time to compute, but I think there is no drone")
            #sys.stdout.write("\ n \r Need time to compute, but I think there is no drone \r \r \r \n")
            #sys.stdout.flush()
            print("Wait for result")
        
        
    #    if sys.stdin in select.select([sys.stdin],[],[],0)[0]:
    #        line = input()
    #        curses.endwin()
    #        break

        i+=1

        #win.refresh()
        #tm.sleep(1)
        #os.system('cls' if os.name == 'nt' else 'clear')
        #win.clear()
        #win.border()
        ##start calculating confidence of occurance

except KeyboardInterrupt:
    pass


print('iter_num:',i)


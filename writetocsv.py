import os, sys, csv
import numpy as np
import soundfile as sf
import librosa

sys.path.append('feature_extraction')

from fextract import *
from filters import *
from lpcgen import *
from harmonics import *

def readaudio(file):
    data, fs = librosa.load(file)
    return data, fs

bandpass = [600, 10000]
nfile_flag = 2
fnid = 909000
for root, dirs, files in os.walk(sys.argv[1]):
    with open(root+'out.csv', 'w',newline='') as f:
        fieldnames = ['ID','Height','Distance','MFCC','CHROMA','MELSPECTROGRAM','SPECTRALCONTRAST','TONNETZ']
        thewriter = csv.DictWriter(f, fieldnames=fieldnames)
        thewriter.writeheader()
        for file in files:
            path = os.path.join(root,file)
            try:
                data, fs = readaudio(path)
            except Exception:
                pass
            ns = bandpass_filter(data,bandpass)
            #try:
            #    p, b = psddetectionresults(data)
            #except IndexError:
            #    pass
            #    b = False
            b = True

            if b:
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(ns, fs)
                #a,e,k = lpc(ns,10)
                #chan,swidth,sfreq,nframes,params,time = print_wave_info(path)
                stri = file[:-4]
                if nfile_flag == 0:
                    with open('recordings1.csv','rt') as f1:
                        reader = csv.DictReader(f1, delimiter=',')
                        for row in reader:
                            fid = row['ID']
                            Height1=row['Height']
                            Distance1=row['Distance']
                            if (stri[:6] == fid):
                                #if not np.isnan(a[1]):
                                thewriter.writerow({'ID':stri,'Height':Height1,'Distance':Distance1,'MFCC':mfccs,'CHROMA':chroma,'MELSPECTROGRAM':mel,'SPECTRALCONTRAST':contrast,'TONNETZ':tonnetz})
                elif nfile_flag == 1:
                    Height1="na"
                    Distance1="na"
                    if not np.isnan(a[1]):
                        thewriter.writerow({'ID':fnid,'Height':Height1,'Distance':Distance1,'MFCC':mfccs,'CHROMA':chroma,'MELSPECTROGRAM':mel,'SPECTRALCONTRAST':contrast,'TONNETZ':tonnetz, 'LPCCOEFF':a[1:],'PERR':e,'RCOEFF':k, 'PSD': p})
                        fnid+=1
                
                elif nfile_flag == 2:
                    Height1=int(stri[:2])
                    Distance1=int(stri[3:5])
                    if True:#not np.isnan(a[1]):
                        thewriter.writerow({'ID':stri,'Height':Height1,'Distance':Distance1,'MFCC':mfccs,'CHROMA':chroma,'MELSPECTROGRAM':mel,'SPECTRALCONTRAST':contrast,'TONNETZ':tonnetz})
                        fnid+=1
				
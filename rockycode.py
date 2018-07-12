#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 04:39:31 2018

@author: sayan
"""

#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import time
import datetime
import os, sys
cdir = os.getcwd()
sys.path.append(cdir)
import graph_functions as gf
import audio_algorithms as aa
import harmonics
from threading import Thread

def record(name = 'output', index = 0, interval = 5, fs=44100):
    print('RECORDING %d TH INTERVAL' % index)
    duration = float(interval)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking = False)
    for i in range(int(interval)):
        sys.stdout.write('\r%d seconds elapsed' % (i+1))
        sys.stdout.flush()
        time.sleep(1)
    recording = recording[:, 0]
    np.save('%s%d' % (name, index), recording)
    return recording

def analyze(recording, all_output):
    print('\nAnalyzing.. ')
    output = harmonics.run(recording)
    aa.print_identify(aa.identify(output))
    all_output.append(output)
    # print(len(all_output))

if __name__ == '__main__':

    if len(sys.argv) < 2:
        sys.exit('not enough arguments, specify overall time and interval time.')
    ovtime = int(sys.argv[1])
    interval = 5
    if len(sys.argv) > 2:
        interval = float(sys.argv[2])
    real_interval = interval
    name = 'output'
    if len(sys.argv) > 3:
        name = sys.argv[3]

    # Start process
    start = datetime.datetime.now()
    print(start.strftime("%Y-%m-%d %H:%M:%S:%f"))
    start_t = time.time()

    deltas = []
    all_output = []
    index = 0
    # Loop
    while(True):
        now = datetime.datetime.now()
        now_t = time.time()
        if now_t - start_t > ovtime:
            break
        print(now.strftime("%Y-%m-%d %H:%M:%S:%f"))
        delta = now - start
        deltas.append(delta)
        recording = record(name = name, index = index, interval = real_interval)
        index += 1
        if index == 1:
            continue
        thread = Thread(target = analyze, args = (recording, all_output, ))
        thread.start()

    thread.join()
    all_output = np.asarray(all_output)
    np.save('%s%s' % (name, str(datetime.datetime.now())), all_output)

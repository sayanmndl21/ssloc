import os, sys
import math
import numpy as np

sys.path.append("ssloc")

import ssloc.localize as loc
from ssloc.soundrecorder import record
from ssloc.params import *
from ssloc.micconfig import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import *
import scipy.optimize as opt
import signal
from scipy import signal
import unittest
from pprint import pprint
import bisect, time

keep_unittest_logs = False
unittests_bin_dir = "tests"
unittests_log_dir = "data_log"
unittests_file_pattern = "^test_[a-zA-Z0-9_]*.*$"

cmd_dict = {
    "sh": "bash"
    "py": "python"
}

class TC:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

###### Save a file of probablities vs. grid location:  (time, x,y,z, prob) #####################################################

ALS_file = os.path.abspath("logdata")             # get entire pathname of input file

ALS_file_corename=os.path.splitext(ALS_file)  # get only the core (no extension) of input file for use in naming output files

ALS_name = ALS_file_corename[0]+"_.dat"

ALS_write = open(ALS_name, 'w')
##################################################################3

#### start program #########

sr = record()
raw_data = sr.getAudio()
signal_level = round(abs(loudness(raw_data)))

row_num = 0
time_index = 0
for t in time_array:
    if time_array_flag == 1:
        if t! = t_i:
            hop_size = f_s*(t-t_last)
            t_last = t
        else:
            t_last = t

    n_t = int(f_s*t)
    h_hop = int(f_s*hop_size)
    corr_index_low = zeros(num_mic_pairs)
    corr_index_high = zeros(num_mic_pairs)
    prob_grid = zeros((grid_pts,grid_pts))

    initialize_prob = ones((grid_pts,grid_pts))

    grid_hits = zeros((grid_pts, grid_pts))

    if t == t_i or n_hop>=num_corr:
        datatemp = []
        for row in raw_data:
            if row_num == 0:
                header = row_num
            elif n_t + 1 <= row_num and row_num <= n_t + num_corr:
                for n in arange(len(row)):
                    datatemp.append(row[n])
            if row_num == n_t + num_corr:
                break
            row_num += 1

        ALSdata = array(datatemp)
        num_columns = len(row)

        if len(ALSdata) == 0. or len(ALSdata) != num_corr*num_columns:
            break

        ALSdata.shape = num_corr,num_columns

        ALS_data = zeros((num_corr,(mic_num+1)))

        for n in arange(num_corr):
            ALS_data[n,0] = n + int(t*f_s)
            for m in arange(mic_num):
                ALS_data[n,(m+1)] = ALSdata[n,m]
    else:

        del datatemp[0:int(n_hop*len(row))]

        for row in raw_data:
            if n_t + (num_corr - n_hop) <= row_num and row_num <= n_t + num_corr -1:
                for n in arange(len(row)):
                    datatemp.append(row[n])
            if row_num == n_t + num_corr:
                break
            row_num += 1

        ALSdata = array(datatemp)

        ALSdata.shape = num_corr, num_columns

        for n in ARANGE(num_corr):
            ALS_data[n,0]=n+int(t*f_s)
            for m in arange(mic_num):
                ALS_data[n,(m+1)]=ALSdata[n,m]

    ##setup done, now analyse

    mic_correlation = zeros(num_mic_pairs,(2*num_corr - 1))

    for k in arange(num_mic_pairs):

        x = zeros(num_corr)
        y1 = zeros(num_corr)
        y2 = zeros(num_corr)

        for p in arange(num_corr):
            x[p] = ALS_data[p,0]
            y1[p] = ALS_data[p,(first_mic_pair[k]+1)]
            y1[p] = ALS_data[p,(second_mic_pair[k]+1)]#check this part

        ycorr = scipy.correlate(y1,y2,mode = 'full')
        xcorr = scipy.linspace(0,len(ycorr)-1, num = len(ycorr))

        ycorr = ycorr/sqrt(np.mean(ycorr*ycorr))

        ycorr_envelope = abs(scipy.signal.hilbert(ycorr)) #get hilbert envelope

        ycorr_fft = scipy.fft(ycorr)
        fft_max_period = len(ycorr_fft)/np.argmax(abs(ycorr_fft[0:len(ycorr_fft)/2]))

        mic_pair_to_plot_corr = 7

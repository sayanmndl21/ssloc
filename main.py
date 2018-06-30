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
import pylab

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
sr.setup()
sr.continuerecord()
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
        if k == mic_pair_to_plot_corr:
            #vizualisedata
            pylab.subplot(311)
            pylab.plot(x,y1,'r.')
            pylab.plot(x,y2,'b.')
            #correlation data
            pylab.subplot(312)
            pylab.plot((xcorr - (len(ycorr)/2)), fabs(ycorr), 'k.')
            pylab.plot((xcorr - (len(ycorr)/2)), ycorr_envelope, 'g.')
            #fft data
            pylab.subplot(313)
            pylab.plot(arange(0,len(ycorr_fft)/2,1),abs(ycorr_fft[0:len(ycorr_fft)/2]))
            pylab.show()

        for i in arange(len(ycorr)):
            mic_correlation[k,i] = ycorr_envelope[i]

        xcorr_center = len(xcorr)/2
        corr_index_max = argmax(ycorr_envelope)
        corr_index_low[k] = int(corr_index_max - 1.5*fft_max_period)
        if corr_index_low[k] <0: corr_index_low[k] = 0
        corr_index_high[k] = int(corr_index_max + 1.5*fft_max_period)
        if corr_index_high[k] > len(ycorr_envelope) -1: corr_index_high[k] = len(ycorr_envelope) -1

    #find lag
    xcorr = xcorr - ones(len(xcorr))*xcorr_center
    for n in arange(grid_pts):
        for m in arange(grid_pts):
            for k in arange(num_mic_pairs):
                p = bisect.bisect(xcorr, del_TOA_mic_pairs[k,n,m]) #search mic correlation matrix for correct distances
                if corr_index_low[k] <= p and p <= corr_index_high[k]:
                    for i in arange(p, corr_index_high[k],1):
                        if fabs(xcorr[i] - del_TOA_mic_pairs[k,n,m])<=box_samples:
                            if add_flag == 0:
                                if initialize_prob[n,m]:
                                    prob_grid[n,m] = 1.
                                    initialize_prob[n,m] = 0
                                prob_grid[n,m] = prob_grid[n,m]*mic_correlation[k,i]#in pressure level
                            else:
                                prob_grid[n,m] = prob_grid[n,m]+mic_correlation[k,i]#in db
                        else:
                            break
                    for i in arange((p-1),corr_index_low[k]-1):
                        if fabs(xcorr[i] - del_TOA_mic_pairs[k,n,m])<=box_samples:
                            if add_flag == 0:
                                if initialize_prob[n,m]:
                                    prob_grid[n,m] = 1.
                                    initialize_prob[n,m] = 0
                                prob_grid[n,m] = prob_grid[n,m]*mic_correlation[k,i]#in pressure level
                            else:
                                prob_grid[n,m] = prob_grid[n,m]+mic_correlation[k,i]#in db
                        else:
                            break
    print("elapsed time to search grid="(time.time()-start),"s")
    #done computing probabilities for this mic pair
    if R_norm_flag == 1:
        for n in arange(grid_pts):
            for m in arange(grid_pts):
                delta_n = n - grid_pts/2
                delta_m = m - grid_pts/2
                if sqrt(delta_n*delta_n + delta_m*delta_m) >= 1.0/box_size:
                    prob_grid[n,m] *= (1.0/box_size)/sqrt(delta_n*delta_n+delta_m*delta_m)

    #rescale
    prob_grid = prob_grid/np.max(prob_grid)
    ###################################################################################################################################
    #localization calculated#view results
    fig = pylab.figure()
    ax = fig.add_subplot(211)
    prob_grid = scipy.transpose(prob_grid)
    if blur_image_flag == 1:
        prob_grid = blur_image(prob_grid, gaussian_kernal)

    prob_grid /= np.max(prob_grid)
    n_pts = len(prob_grid)

    pylab.imshow(prob_grid)
    pylab.grid(True)

    #colorbar
    pylab.colorbar()
    pylab.xlabel('x')
    pylab.ylabel('y')
    pylab.title('pixel width = '+str(box_size)+' meter')

    #label mics
    mic_x = array([mic_array_x[0]/box_size+n_pts/2,mic_array_x[1]/box_size+n_pts/2)
    mic_y = array([mic_array_y[0]/box_size+n_pts/2,mic_array_y[1]/box_size+n_pts/2)
    pylab.plot(mic_x[:],mic_y[:],'r.')
    pylab.annotate("1", (mic_array_x[0]/box_size+n_pts/2,mic_array_y[0]/box_size+n_pts/2),fontsize=12,color='red')
    pylab.annotate("2", (mic_array_x[1]/box_size+n_pts/2,mic_array_y[1]/box_size+n_pts/2),fontsize=12,color='red')

    #view location
    plot_angle = ogrid[-pi:pi+0.05*pi:0.05*pi]
    x = (3.89/box_size)*cos(plot_angle)+n_pts/2
    y = (3.89/box_size)*sin(plot_angle)+n_pts/2
    pylab.plot(x[:],y[:],color='white')

    block_size = int(n_pts/num_blocks)
    std_value = np.std(prob_grid)

    lm_counter = 0#local maxima
    angle_lm_mean = 0

    if plot_local_maxima == 1:
        for n in arange(0,n_pts, block_size):
            for m in arange(0, n_pts, block_size):
                min_value = np.min(prob_grid[n:block_size,m:m+block_size])
                max_value = np.max(prob_grid[n:block_size,m:m+block_size])
                if fabs(max_value) > fabs(min_value) + num_stdev*std_value:
                    block_max = np.argmax(prob_grid[n:n+block_size,m:m+block_size])
                    n_plt = n + block_max/block_size
                    m_plt = m + block_max%block_size
                    if m_plt > n_pts/2:
                        text_x = m_plt -2
                    else:
                        text_x = m_plt + 1
                    if n_plt > n_pts/2:
                        text_y = n_plt - 2
                    else:
                        text_y = n_plt +4

                    pylab.annotate('('+str(m_plt)+','+str(n_plt)+')', xy=(m_plt, n_plt), xytext=(text_x,text_y))
                    pylab.plot([m_plt],[n_plt],'wo',markersize=4)
                    print("local max (x,y)=(%.2f,%.2f,%.2f)" % ((m_plt-n_pts/2)*box_size,(n_plt-n_pts/2)*box_size,max_value))

                    #compute angle wrt x_axis
                    r_lm = vector((m_plt - n_pts/2),(n_plt - n_pts/2))
                    if r_lm.y > 0:
                        angle_lm = arccos(dot(norm(r_lm),(1,0,0)))
                    else:
                        angle_lm = -1.*arccos(dot(norm(r_lm),(1,0,0)))
                    angle_lm_mean += angle_lm
                    lm_counter += 1
                    if beanforming != 1:
                        ALS_write.write((str(t)+" "+str((m_plt-n_pts/2)*box_size)+" "+str((n_plt-n_pts/2)*box_size)+" "+str(max_value)+"\n") # save local maxima

    ax.set_xlim(0,n_pts)
    ax.set_ylim(0,n_pts)
    ax.set_xticks(arange(0,n_pts,block_size))
    ax.set_yticks(arange(0,n_pts,block_size))
    pylab.savefig(ALS_file_corename[0]+"_"+str(time_index)+".png")

    global_max = np.argmax(prob_grid)
    n_plt = global_max/n_pts
    m_plt = global_max%n_pts
    pylab.plot([m_plt],[n_plt],'wx',markersize = 5)

    print("global maxima at [x,y]=",m_plt,n_plt, ((m_plt - n_pts/2)*box_size,(n_plt-n_pts/2)*box_size))
    if lm_counter != 0:
        angle_lm_mean /= lm_counter
    print("mean angle=", angle_lm_mean*360./(2.*pi))

    if beamforming == 1:
        ALS_write.write(str(t)+" "+str(angle_lm_mean*360./(2.*pi))+"\n")

    pylab.draw()

    time_index += 1
    print("white dots = local maxima in regions ,",block_size*box_size," meter square")

ALS_write.close()
pylab.show()

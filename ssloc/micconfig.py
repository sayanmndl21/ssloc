import math, time
import numpy as np
from numpy import *
from params import *
from vpython import *

mic_num = 2
num_mic_pairs = int(math.factorial(mic_n)/(math.factorial(mic_n-2)*2))

#mic array coordinates
mic_array_x = np.zeros(mic_n)
mic_array_y = np.zeros(mic_n)
mic_array_z = np.zeros(mic_n)

mic_array_x[0] = -0.28 # update array setup to actual setup
mic_array_x[1] = 0.28
mic_array_y[0] = 0.05
mic_array_y[1] = 0.04

for n in np.arange(mic_num):
    mic_array_z[n] = 0

start = time.time()
grid_mic_dist=np.zeros( (mic_num,grid_pts,grid_pts))
                                    # this array holds the distances from every grid point to every microphone, computed once and for all up front
for k in np.arange(mic_num):
    mic_vector = vector(mic_array_x[k],mic_array_y[k],mic_array_z[k])
    for n in np.arange(grid_pts):
        for m in np.arange(grid_pts):
            grid_box_vector = vector(  box_size*(n - grid_pts/2)  ,  box_size*(m - grid_pts/2), 0  )
            grid_mic_dist[k,n,m]= mag(grid_box_vector - mic_vector )

# now compute the array of delta_distances to each mic from each grid box--this is what the difference in micrphone time of arrival measures

del_TOA_mic_pairs=np.zeros((int(num_mic_pairs),int(grid_pts),int(grid_pts))) # array that contains the difference in TOA from each grid point to the mics in a pair

mic_pairs_bookkeeping = np.zeros((mic_num,mic_num))        # this ensures we only count each permutation once
n_pair = 0                                              # a counter that keeps track of which pair we are considering

first_mic_in_pair = np.zeros(num_mic_pairs)                # initialize arrays that hold the identify of the various microphones in each pair of the num_mic_pairs pairs
second_mic_in_pair = np.zeros(num_mic_pairs)
angle_mic_pair = np.zeros(num_mic_pairs)                   # initialize array that holds the angle between +y-axis and the (i,j) mic pair vector (ri-rj)

mic_pair_center_x = np.zeros(num_mic_pairs)                # coordinates of the center of each mic array
mic_pair_center_y = np.zeros(num_mic_pairs)
mic_pair_vector_x = np.zeros(num_mic_pairs)                # coordinates of the vector from i to j in each (i,j) pair
mic_pair_vector_y = np.zeros(num_mic_pairs)

for k in arange(mic_num):
    for l in arange(mic_num):
        if k != l and not (mic_pairs_bookkeeping[k,l]): # this ensures we only count each permutation once
            mic_pairs_bookkeeping[k,l] = 1              # this ensures we only count each permutation once
            mic_pairs_bookkeeping[l,k] = 1              # this ensures we only count each permutation once
            for n in arange(grid_pts):
                for m in arange(grid_pts):              # compute the difference in times of arrival (TOAs) in units of samples
                    del_TOA_mic_pairs[n_pair,n,m] = (grid_mic_dist[k,n,m]-grid_mic_dist[l,n,m])/(v_sound*dt)  # if > 0 farther from 1st mic k than from 2nd mic
            first_mic_in_pair[n_pair] = k
            second_mic_in_pair[n_pair] = l
                                                        # now compute the angle the (k,l)mic pair  makes with the +y-axis
            angle_mic_pair[n_pair]=math.acos( norm(vector(mic_array_x[k],mic_array_y[k],mic_array_z[k])-vector(mic_array_x[l],mic_array_y[l],mic_array_z[l])).y )
            mic_pair_center_x[n_pair] = 0.5*(mic_array_x[k]+mic_array_x[l])
            mic_pair_center_y[n_pair] = 0.5*(mic_array_y[k]+mic_array_y[l])
            mic_pair_vector_x[n_pair] =  mic_array_x[k]- mic_array_x[l]
            mic_pair_vector_y[n_pair] =  mic_array_y[k]- mic_array_y[l]

            n_pair = n_pair + 1

mic_spacing = zeros(num_mic_pairs)
lag_max = zeros(num_mic_pairs)
for k in arange(num_mic_pairs):
    dx = mic_array_x[int(first_mic_in_pair[k])] - mic_array_x[int(second_mic_in_pair[k])]
    dy = mic_array_y[int(first_mic_in_pair[k])] - mic_array_y[int(second_mic_in_pair[k])]
    dz = mic_array_z[int(first_mic_in_pair[k])] - mic_array_z[int(second_mic_in_pair[k])]
    mic_spacing[k]=sqrt(dx*dx+dy*dy+dz*dz)
    lag_max[k] = int((mic_spacing[k] + box_size)*f_s/v_sound)   # number of lag times (in samples) we can reasonably go around zero before it stops making sense physically
                                                                # because the two channel's sounds could not correspond to the same source
max_lag_time = max(lag_max)                             # find the maximum lag time that can result for any of the microphone pairs, to use when computing correlation time windows

win_cor = 1.2*( max_lag_time/f_s + signal_duration) # time window in sec for computing the cross-correlation functions:  try for the call length in s + >20% silence on either end
num_corr = int(f_s*win_cor)                         # the total number of data points in each correlation function input function (only the number of samples we used to compute correlations;  the waveform has more
                                                    # for analyzing the nitro popper file as an example, I used num_corr = 75, t= int(9.2315 * f_s) below, box_size = 0.1, grid_pts = 100

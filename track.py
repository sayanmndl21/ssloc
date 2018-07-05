import numpy as np
import math
from visual import *
from visual.graph import * # import graphing features
from visual.filedialog import get_file
import scipy
from scipy import signal
import pylab
import scipy.optimize
import os.path
import csv
import bisect
import time

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    # 3 x 3 array with sigma = sqrt (n/2)
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im, g, mode='valid')
    return(improc)

######## parameter initialization #####################################################################

# The size limit for .csv files is pretty modest.  To see how big, try print csv.field_size_limit()
csv.field_size_limit(1000000000)    # Make it larger so we can read in big sound files!

# plotting & ALS analysis
beamforming = 0                     # *** if your array is linear or if your microphone spacings are much smaller than the distances
                                    # *** to your sounds sources, you should set this = 1, which will result in a computation of only
                                    # *** the average angle to the sound source;  inaccuracies in alignment make it impossible to do more than
                                    # *** beamforming (angle determination) for a single array of small size;  multiple arrays separated
                                    # *** by a large distance can resolve the actual position by trilateration, or you can use a microphone
                                    # *** array with a large spacing compared to your sound source distances for true 2D localization
                                    # *** If you wish full 2D localization, set = 0
R_norm_flag = 0	                    # *** normalize probabilities by 1/R factor? The references don't recommend; this would reduce the weighting per grid
                                    # *** box by 1/(distance from array center) to account for size of angular size of grid boxes diminishing with distance-??? correct???
add_flag = 1                        # *** 0 = multiply correlation values to get join probability;  1 = add them as in filter-and-sum or if in dB
plot_local_maxima = 1               # *** search for local maxima
num_stdev = 2.5                     # *** by how much greater (in standard deviations) a local maximum must be from the minimum to get noted
num_blocks = 8                      # *** when searching for local maxima, we divide the ALS grid into this many blocks per edge
blur_image_flag = 1                 # *** uses a Gaussian kernel to blur the image?
gaussian_kernel = 2	            # *** size (in grid boxes) of Gassian blurring kernel:  sigma = 1.22 * this value
# end of initializing plotting final ALS data

v_sound = 350                       # *** speed of sound (m/s) -- be sure to input since it's a function of temp. & relative humidity
f_s = 44100                         # *** sampling frequency (Hz)
dt = 1/float(f_s)                   # our sampling time interval
del_x = dt*v_sound                  # the smallest discernable difference in distance between the sound source and the microphone array
bandwidth = 5e3                     # *** our bird call's frequency bandwidth in Hz <-- used in uncertainty principle computation of box size
signal_duration = 0.11              # *** the duration (s) of the signals of interest (for example, the duration of each syllable or trill in a bird call)
                                    # this is a typical value for our tree swallow notes
hop_size = 0.11                     # *** increment in time (s) over which to move in computing next value of the localization position
if hop_size > signal_duration:
    hop_size = signal_duration
    print "you set the hop size too small--the software isn't set up to handle steps in time greater than the correlation window"



time_array_flag = 1                 # *** set = 1 if we use an array of discrete times to loop over in our animation (see below) rather than using a time index
                                    # *** set = 0 if you wish to use t_i, t_f and hop_size, as above
                                    # and starting and stopping times, t_i and t_f and hop_size

if time_array_flag == 1:
    time_array = [15.213,51.0,70.603,111.806,156.397,184.619,252.988,290.141,319.861] # times are stored in a starting array like this one
    t_i = time_array[0]                               # set initial time here from array entries
    hop_size = f_s*(time_array[1] -time_array[0])   # redefine hop_size as # samples between each time in array
else:
    t_i= 15.213                     # *** starting time
    t_f= 70.603                     # *** ending time
    t_f= t_f + hop_size
    time_array = arange(t_i,t_f,hop_size)
                                        # sets up the time to analyze array using the hop_size,start and stop times
#### set up the spatial grid that we do our computations on ##############################################################
grid_pts = 100                      # *** number of boxes across our square grid
box_size = 0.200                    # *** edge length of each grid box in meters
if v_sound/(2.*3.14159*bandwidth) <= box_size:
    print "Each grid box is ",box_size," m to an edge, which is >= ",v_sound/(2.*3.14159*bandwidth), "the smallest allowable box_size from the uncertainty principle"
else:
    print "Your grid box size was too big; it has been set to ",v_sound/(2.*3.14159*bandwidth), "the smallest allowable box_size from the uncertainty principle"
    box_size = v_sound/(2.*3.14159*bandwidth) # size of each box edge should be >= our resolution in distances from the cross-correlations
grid_size = box_size*grid_pts                 # width of field of view along x, y and z directions;  10 meter for now -- about right for the distance to our sound sources at most!

box_samples = box_size/(2.*v_sound*dt)        # an array that holds the number of samples that correspond to the travel time for each box_size/2; used to match grid boxes to del TOA's

############ set up microphone array parameters ###########################################################################
mic_num = 4
# the list of permutations is defined as: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
num_mic_pairs = factorial(mic_num)/(factorial(mic_num - 2) * 2)   # = 6 for 4 mic's:  the number of possible mic pairs to consider

# get microphone array coordinates
mic_array_x = zeros(mic_num)
mic_array_y = zeros(mic_num)
mic_array_z = zeros(mic_num)

# input microphone array information *** if you wish you can uncomment out this part and enter values by hand
##for n in arange(mic_num):
##    mic_loc = input("Enter Mic "+str(n)+" x(in m): ")   # input by hand the real world coords. of the array microphones
##    mic_array_x[n] = float(mic_loc)
##    mic_loc = input("Enter Mic "+str(n)+" y(in m): ")   # input by hand the real world coords. of the array microphones
##    mic_array_y[n] = float(mic_loc)
# now draw in microphone array in actual position on the grid

##Square array microphone positions (meters):
##

mic_array_x[0] = -0.28 # 7-15-11 square array coordinates
mic_array_x[1] = 0.28
mic_array_x[2] = -0.08
mic_array_x[3] = -0.08
mic_array_y[0] = 0.05
mic_array_y[1] = 0.04
mic_array_y[2] =  -0.27
mic_array_y[3] =  0.30

for n in arange(mic_num):             # *** for a square array, set the z-values equal to zero
    mic_array_z[n] = 0

##for n in arange(mic_num):
##    mic_array_y[n] = 0              # *** y & z are 0 for linear arrays
##    mic_array_z[n] = 0
######## end parameter initialization #################################################################

######## Setup microphone array #######################################################################
start = time.time()

# compute the distances from each grid point to each microphone
grid_mic_dist=zeros( (mic_num,grid_pts,grid_pts))
                                    # this array holds the distances from every grid point to every microphone, computed once and for all up front
for k in arange(mic_num):
    mic_vector = vector(mic_array_x[k],mic_array_y[k])
    for n in arange(grid_pts):
        for m in arange(grid_pts):
            grid_box_vector = vector(  box_size*(n - grid_pts/2)  ,  box_size*(m - grid_pts/2)  )
            grid_mic_dist[k,n,m]= mag(grid_box_vector - mic_vector )

# now compute the array of delta_distances to each mic from each grid box--this is what the difference in micrphone time of arrival measures

del_TOA_mic_pairs=zeros((num_mic_pairs,grid_pts,grid_pts)) # array that contains the difference in TOA from each grid point to the mics in a pair

mic_pairs_bookkeeping = zeros((mic_num,mic_num))        # this ensures we only count each permutation once
n_pair = 0                                              # a counter that keeps track of which pair we are considering

first_mic_in_pair = zeros(num_mic_pairs)                # initialize arrays that hold the identify of the various microphones in each pair of the num_mic_pairs pairs
second_mic_in_pair = zeros(num_mic_pairs)
angle_mic_pair = zeros(num_mic_pairs)                   # initialize array that holds the angle between +y-axis and the (i,j) mic pair vector (ri-rj)

mic_pair_center_x = zeros(num_mic_pairs)                # coordinates of the center of each mic array
mic_pair_center_y = zeros(num_mic_pairs)
mic_pair_vector_x = zeros(num_mic_pairs)                # coordinates of the vector from i to j in each (i,j) pair
mic_pair_vector_y = zeros(num_mic_pairs)

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
            angle_mic_pair[n_pair]=arccos( norm(vector(mic_array_x[k],mic_array_y[k])-vector(mic_array_x[l],mic_array_y[l])).y )
            mic_pair_center_x[n_pair] = 0.5*(mic_array_x[k]+mic_array_x[l])
            mic_pair_center_y[n_pair] = 0.5*(mic_array_y[k]+mic_array_y[l])
            mic_pair_vector_x[n_pair] =  mic_array_x[k]- mic_array_x[l]
            mic_pair_vector_y[n_pair] =  mic_array_y[k]- mic_array_y[l]

            n_pair = n_pair + 1

mic_spacing = zeros(num_mic_pairs)
lag_max = zeros(num_mic_pairs)
for k in arange(num_mic_pairs):
    dx = mic_array_x[first_mic_in_pair[k]] - mic_array_x[second_mic_in_pair[k]]
    dy = mic_array_y[first_mic_in_pair[k]] - mic_array_y[second_mic_in_pair[k]]
    dz = mic_array_z[first_mic_in_pair[k]] - mic_array_z[second_mic_in_pair[k]]
    mic_spacing[k]=sqrt(dx*dx+dy*dy+dz*dz)
    lag_max[k] = int((mic_spacing[k] + box_size)*f_s/v_sound)   # number of lag times (in samples) we can reasonably go around zero before it stops making sense physically
                                                                # because the two channel's sounds could not correspond to the same source
max_lag_time = max(lag_max)                             # find the maximum lag time that can result for any of the microphone pairs, to use when computing correlation time windows

print "done computing microphone array parameters"
print "time for processing microphone array data = ", (time.time() - start) , " s"

#######################################################################################################################
# Set up the correlation parameters ###############################################################################
# num_corr = record size (number of samples used for computing the actual correlation functions) in Raven Pro beamforming

win_cor = 1.2*( max_lag_time/f_s + signal_duration) # time window in sec for computing the cross-correlation functions:  try for the call length in s + >20% silence on either end
num_corr = int(f_s*win_cor)                         # the total number of data points in each correlation function input function (only the number of samples we used to compute correlations;  the waveform has more
                                                    # for analyzing the nitro popper file as an example, I used num_corr = 75, t= int(9.2315 * f_s) below, box_size = 0.1, grid_pts = 100

# make sure that the win_corr used is at least as long as the maximum mic spacing/v_sound from maximum lag_time so you can get contributions from all angles

# Done setting up the correlation parameters ###############################################################################

######################################################################################################################

# file i/o section:  read in the .csv file created by Raven from our .WAV files
#
# In Raven, make the waveform window active, then go to File/Export Sound Samples... and save as a .csv file
# These files contain no time information apart from the row number (you need to convert using the sampling frequency,
#   which is available in Raven under the file's information);  the file is just a 4-column spreadsheet of the 4 channels' sound waveforms
# and digest it to get (time, Ch1, Ch2, Ch3, Ch4) waveform
# Be sure to prefilter files in Raven Pro 1.4 before analyzing them with this program.  For our swallow mobbing cries, put them through
# the Tools/Batch Band Filter using the Band Pass Filter from 1.5 KHz to 9kHz.  You need to create a directory for the input and output files
# for a batch process, so be careful not to overwrite the original files!  Failing to bandpass filter the files will widen the cross-correlations
# due to the much lower signal-to-noise ratio for unfiltered files.  For other calls, be sure to get the correct bandpass filter values.
#

print "Get multichannel sound data now (as .CSV file)"

fd = get_file()

print fd.name

start = time.time()

ifile  = open(fd.name, "rb")
reader = csv.reader(ifile)                      # reader = object that iteratures over lines in the .csv file being read in;  this is an ongoing process
reader_length = reader.line_num                 # the number of lines read in from the file

print "time to read in multichannel sound file = ", (time.time() - start) , " s"

###### Save a file of probablities vs. grid location:  (time, x,y,z, prob) #####################################################

ALS_file = os.path.abspath(fd.name)             # get entire pathname of input file

ALS_file_corename=os.path.splitext(ALS_file)  # get only the core (no extension) of input file for use in naming output files

ALS_name = ALS_file_corename[0]+"_.dat"

ALS_write = open(ALS_name, 'w')
######## MAIN LOOP ######################################################################################################

row_num = 0                                         # this counter keeps track of where we are in our sound file
time_index = 0
for t in time_array:                                # loop over all times in input array
    print "working on time ",t," s"
    if time_array_flag == 1:                        # redefine hop_size each step when we use a discrete array of times only
        if t!=t_i :
            hop_size = f_s*(t-t_last)               # redefine hop_size each time if you use a discrete time array
            t_last = t                              # set t_last as the last time encountered in the loop
        else:
            t_last = t                              # initialize t_last

    start = time.time()                             # start timer for how long each section takes to execute

    n_t = int(f_s*t)                                # this is the number of samples corresponding to time t
    n_hop = int(f_s*hop_size)                       # the number of samples in a hop size
    # initialize all necessary values
    corr_index_low = zeros(num_mic_pairs)           # indices that surround the highest cross-correlation maximum for each mic pair
    corr_index_high = zeros(num_mic_pairs)
    prob_grid = zeros((grid_pts,grid_pts))          # initialize array of probabilities at each grid point--zero to start with

    initialize_prob=ones((grid_pts,grid_pts))       # initialize an array of flags that tells us whether to initialize the probability in each grid to 1 or  not--do only once for the entire matrix,
                                                    # outside of the mic pair number loop, but within the time loop
    grid_hits=zeros((grid_pts,grid_pts))            # array to keep track of how many matches there are between the correlations and each grid box

    # read in values of data for this region of time
    if t == t_i or n_hop >= num_corr:               # for the first pass, store num_corr values and set up all initialization of arrays, lengths, etc.
        datatemp=[]                                 # clear & initialize dummy array for handling file i/o
        for row in reader:
            if row_num == 0:                        # Save header row.
                header = row
            elif n_t+1 <= row_num  and row_num <= n_t + num_corr:        # the +1 is because t = 0 gives n_t = 0, but row 0 is the header
                for n in arange(len(row)):          # if the data is within our interval of interest, then store it for analysis
                    datatemp.append(row[n])
            if row_num == n_t + num_corr:           # stop here in the file--at the end of the region to analyze this time
                break
            row_num += 1                            # keep track of where you are in file:  this is # samples, starting at entry 1 (entry 0 is the header file)

        ALSdata = array(datatemp)

        num_columns = len(row)                          # number of columns should be number in each row;  rownum = # rows + 1

        if len(ALSdata) == 0. or len(ALSdata)!= num_corr*num_columns:   # if we are at or beyond the end of the file, say so and break
            print "trying to analyze a time beyond the file's end"
            break

        ALSdata.shape = 3072,num_columns            # reshape it into a 2D array of the right size

        ALS_data = zeros((num_corr,(mic_num+1)))        # initialize actual data array
        # our data array datatemp consists of strings of numbers;  each row holds "ch1","ch2","ch3","ch4"," "; where of course the first 4 entries are
        #  the multichannel sound data for 4 channels; there is always a blank at the end of each line, so len(row) = 5;  we ignore the blanks below
        for n in arange(num_corr):                      # make up actual data array
            ALS_data[n,0]=n + int(t*f_s)                # time in terms of number of samples
            for m in arange(mic_num):
                ALS_data[n,(m+1)]=ALSdata[n,m]          # now store all the channel of the waveform properly
                                                        # setup of this array:    ALS_data[n,0] = time in samples
                                                        #                         ALS_data[n,i] = channel i's waveform value at time determined by n
    else:                                               # for all times not equal to the start time

        del datatemp[0:int(n_hop*len(row))]             # first, remove the first hop_size entries from data_temp <--how many to remove?

        for row in reader:                              # this ought to read in and append just n_hop (# samples in a hop size) more rows of data
            if n_t + (num_corr - n_hop) <= row_num  and row_num <= n_t + num_corr -1 :        # the +1 is because t = 0 gives n = 0, but row_num=0 is the header
                                                        # after the first time, only read the next hop_size samples and append to the list
                for n in arange(len(row)):              # if the data is within our interval of interest, then store it for analysis
                    datatemp.append(row[n])             # here it's appending len(row) entries!!! we have to pop len(row)*n_hop entries at start
            if row_num == n_t + num_corr:                  # stop here in the file--at the end of the region to analyze this time
                break
            row_num += 1                                # keep track of where you are in file:  this is # samples, starting at entry 1 (entry 0 is the header file))

        ALSdata = array(datatemp)

        ALSdata.shape = num_corr,num_columns            # reshape it into a 2D array of the right size

        for n in arange(num_corr):                      # make up actual data array
            ALS_data[n,0]=n + int(t*f_s)                # time in terms of number of samples
            for m in arange(mic_num):
                ALS_data[n,(m+1)]=ALSdata[n,m]          # now store all the channel of the waveform properly

    # done reading in num_corr data points

    # Now set up the main loop for analysis.  For each microphone pair:  1) compute correlations;  2) for each correlation function lag time,
    #   find matches in grid and insert probabilities where each match occurs.
    # Once complete:  compute & plot final probability map and compute center of mass of source of sound (if only one)
    mic_correlation = zeros((num_mic_pairs,(2*num_corr-1))) # a matrix that holds all the cross-correlations

    for k in arange(num_mic_pairs):                     # loop over all permuted pairs of mic's (see above for details)

        x = zeros(num_corr)                         # dimension each array in advance now;  these are used to compute the cross-correlations
        y1 = zeros(num_corr)
        y2 = zeros(num_corr)

        for p in arange(num_corr):
            x[p] = ALS_data[p,0]                                # the window in time over which to do the correlation
            y1[p] = ALS_data[p,(first_mic_in_pair[k]+1)]        # 1st channel in pair
            y2[p] = ALS_data[p,(second_mic_in_pair[k]+1)]       # 2nd channel in pair

        # Notes
        # 1)  For a narrowband signal, there is an oscillation in the cross-correlation function at that freq.
        # Then spatial aliasing occurs because this shows up as modulations of the reconstructed position
        # 2)  Looking at these plots, we see that negative xcorr (the lag) values mean that the sound source is closer to the first mic in the pair
        #     than to the second mic in the pair
        #  (i.e., that the corresponding sign of del_TOA_mic_pairs should be negative as well

        # compute the cross-correlation between y1 and y2
        ycorr = scipy.correlate(y1, y2, mode='full')
        xcorr = scipy.linspace(0, len(ycorr)-1, num=len(ycorr))

        ycorr = ycorr/sqrt(np.mean(ycorr*ycorr))                # normalize ycorr so it too doesn't cause too big a final prob_grid values
                                                                # with our closely spaced mic's and our closely matched mic gains, not a big deal

        ycorr_envelope = abs(scipy.signal.hilbert(ycorr))       # this computes the Hilbert envelope, as recommended by the UCLA group to avoid
                                                                # spatial aliasing due to rapid oscillations in the cross-correlation function and
                                                                # the many zeros that imposes--alternative or additional:  square then, take low-pass
                                                                # filter, then sqrt to remove rapid beat frequencies.
        y_corr_fft = scipy.fft(ycorr)
        fft_max_period = len(y_corr_fft)/np.argmax(abs(y_corr_fft[0:len(y_corr_fft)/2]))
                                                                # find the peak of this fft--use to define how far around our maximum we will use ***
                                                                # for an FFT, the length = f_s / peak frequency = period in in terms of original ycorr index
##        print "period (argument) of the fft maximum = ",fft_max_period,np.argmax(abs(y_corr_fft[0:len(y_corr_fft)/2]))

        mic_pair_to_plot_corr = 7                               # set > 5 if you don't wish to view the cross-correlation functions; = k if you do
        if k == mic_pair_to_plot_corr:                          # allows plotting the correlation functions for whichever pair I like

            # visualize the data <-- IMPORTANT NOTE:  the program pauses while the first window is open; just close the plot window and it will continue
            # Use this section to look at your data to be sure that you have used a wide enough window for the correlation calculations to encompass
            # both signals for all the mic pairs!
            # plot the initial functions
            pylab.subplot(311)
            pylab.plot(x, y1, 'r.')
            pylab.plot(x, y2, 'b.')

            # plot the correlation
            pylab.subplot(312)
            pylab.plot((xcorr-(len(ycorr)/2) ), fabs(ycorr), 'k.')      # plot the absolute value of the correlation function
            pylab.plot((xcorr-(len(ycorr)/2) ), ycorr_envelope, 'g.')   # plot its Hilbert envelope
            pylab.subplot(313)

            pylab.plot(arange(0,len(y_corr_fft)/2,1),abs(y_corr_fft[0:len(y_corr_fft)/2]))   # plot its FFT
            pylab.show()

        for i in arange(len(ycorr)):
            mic_correlation[k,i]=  ycorr_envelope[i]            # use the Hilbert envelope (amplitudes) of the cross-correlations

        xcorr_center = len(xcorr)/2                             # location of zero lag time <--I checked that this works for odd lengths also
        corr_index_max = argmax(ycorr_envelope)                      # get the index of maximum of the cross-correlation
        corr_index_low[k] = int(corr_index_max  - 1.5*fft_max_period)         # compute the range of lag time indices to use in including the maximum correlation peak
        if corr_index_low[k] < 0:  corr_index_low[k] = 0
        corr_index_high[k] = int(corr_index_max  + 1.5*fft_max_period)
        if corr_index_high[k] > len(ycorr_envelope) -1 :  corr_index_high[k] =  len(ycorr_envelope)- 1

    # now, use xcorr = lag index to get lag time = difference in time of arrival (TOA) = xcorr/f_s to compute differences in distance from source to each microphone
    # loop over all grid points, then all microphone pairs, searching for the ones that agree with this distance
    # if the TOA distances for each microphone pair agree w/in uncertainty with corresponding distances for each grid point
    # add cross-correlation for that pair's TOA distance to that grid point's

    # start the loop that assigns probabilies to grid boxes for this mic pair here #####

    xcorr = xcorr - ones(len(xcorr))*xcorr_center               # correct so it's centered at zero lag time
    for n in arange(grid_pts):
        for m in arange(grid_pts) :
            for k in arange(num_mic_pairs):

                p = bisect.bisect(xcorr,del_TOA_mic_pairs[k,n,m])   # search mic correlation matrix for the correct value of distance (in units of samples)
                                                                    # this is the value of p to use in searching the correlation matrix for matches
                if corr_index_low[k] <= p and p <= corr_index_high[k]:               # only search if it's within range of the central maximum
                    for i in arange(p,corr_index_high[k],1):                    # search upward from p to len(xo)
                        if fabs(xcorr[i] - del_TOA_mic_pairs[k,n,m]) <= box_samples:
                            if add_flag == 0:
                                if initialize_prob[n,m]:
                                    prob_grid[n,m] = 1.                     # the first time each [n,m] point's del_dist to mic pair matches a lag_time value, we set to one
                                    initialize_prob[n,m]=0                  # and set this flag to zero so it only happens once
                                prob_grid[n,m] = prob_grid[n,m] * mic_correlation[k,i]    # multiply the correlation amplitude to get the joint probability--add if in dB
                            else:
                                prob_grid[n,m] = prob_grid[n,m] + mic_correlation[k,i]    # add the correlation amplitude to get the joint probability
                        else:
                            break
                    for i in arange((p-1),corr_index_low[k],-1):                        # search downward from p to 0
                        if fabs(xcorr[i] - del_TOA_mic_pairs[k,n,m]) <= box_samples:
                            if add_flag == 0:
                                if initialize_prob[n,m]:
                                    prob_grid[n,m] = 1.                     # the first time each [n,m] point's del_dist to mic pair matches a lag_time value, we set to one
                                    initialize_prob[n,m]=0                  # and set this flag to zero so it only happens once
                                prob_grid[n,m] = prob_grid[n,m] * mic_correlation[k,i]    # multiply the correlation amplitude to get the joint probability--add if in dB
                            else:
                                prob_grid[n,m] = prob_grid[n,m] + mic_correlation[k,i]    # add the correlation amplitude to get the joint probability
                        else:
                            break

    print "elapsed time to search grid & store correlations = ", (time.time() - start), " s"
    #### done computing the probablities for this mic pair
    if R_norm_flag == 1:
        for n in arange(grid_pts):                          # rescale the values by 1/R to take into account the fact that each grid_box shrinks in
            for m in arange(grid_pts) :                     # effective intercepted angle with distance
                delta_n = n - grid_pts/2
                delta_m = m - grid_pts/2
                if sqrt(delta_n*delta_n+delta_m*delta_m) >= 1.0/box_size:
                    prob_grid[n,m] *= (1.0/box_size)/sqrt(delta_n*delta_n+delta_m*delta_m)

    # rescale the maximum prob_grid to one for plotting
    prob_grid=prob_grid/np.max(prob_grid)  # this sets the maximum value in each grid box to 1<--needed to keep values under control later
    ###### End of localization calculations ##########################################################################
    # plot using matplotlib and try smoothing #########################################################################


    fig = pylab.figure()                            # open and name the figure
    ax = fig.add_subplot(211)                       # naming the plot allows us to set its axis ranges later
    prob_grid=scipy.transpose(prob_grid)            # switch x and y axes for display
    if blur_image_flag == 1:
        prob_grid = blur_image(prob_grid, gaussian_kernel)        # smoothes with a gaussian_kernel x gaussian_kernel Gaussian filter,
                                                    # with with sigma = sqrt (3/2) ~ 1.22 grid boxes
    prob_grid /= np.max(prob_grid)                  # rescale again for ease of reading colormaps
    n_pts = len(prob_grid)                          # the new # points after blurring <-- we lose some points at the edges

    pylab.imshow(prob_grid)
    pylab.grid(True)

    # draw in a colorbar for intensity/probability scale
    pylab.colorbar()
    pylab.xlabel('x')
    pylab.ylabel('y')
    pylab.title('pixel width = '+str(box_size)+' meter')

    # draw in the labels where each mic is see  http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.close
    mic_x = array([mic_array_x[0]/box_size+n_pts/2,mic_array_x[1]/box_size+n_pts/2,mic_array_x[2]/box_size+n_pts/2,mic_array_x[3]/box_size+n_pts/2])
    mic_y = array([mic_array_y[0]/box_size+n_pts/2,mic_array_y[1]/box_size+n_pts/2,mic_array_y[2]/box_size+n_pts/2,mic_array_y[3]/box_size+n_pts/2])
    pylab.plot(mic_x[:],mic_y[:],'r.')
    pylab.annotate("1", (mic_array_x[0]/box_size+n_pts/2,mic_array_y[0]/box_size+n_pts/2),fontsize=12,color='red')
    pylab.annotate("2", (mic_array_x[1]/box_size+n_pts/2,mic_array_y[1]/box_size+n_pts/2),fontsize=12,color='red')
    pylab.annotate("3", (mic_array_x[2]/box_size+n_pts/2,mic_array_y[2]/box_size+n_pts/2),fontsize=12,color='red')
    pylab.annotate("4", (mic_array_x[3]/box_size+n_pts/2,mic_array_y[3]/box_size+n_pts/2),fontsize=12,color='red')

    # *** for 7-5-11 test data only, draw in a circle that's at the right location for our sources:
    plot_angle= ogrid[-pi:pi+0.05*pi:0.05*pi]
    x = (3.89/box_size)*cos(plot_angle)+n_pts/2
    y = (3.89/box_size)*sin(plot_angle)+n_pts/2
    pylab.plot(x[:], y[:],color='white')
    # *** end of drawing circle for 7-5-11 test data only

    # new loop--divide into small regions, find local maximum, then if above 2 sigma, plot annotation where each is:
    # xy (arrow tip) and xytext locations (text location) are in data coordinates.

    block_size = int(n_pts/num_blocks)                  # how many grid boxes per edge are in a block to search for local maxima

    std_value = np.std(prob_grid)                       # get a single value for the entire grid

    lm_counter = 0                                      # number of local maxima detected
    angle_lm_mean = 0                                   # average angle (w.r.t. +x-axis) for the local maxima

    if plot_local_maxima == 1:
        for n in arange(0,n_pts,block_size):
            for m in arange(0,n_pts,block_size):
                min_value = np.min(prob_grid[n:n+block_size,m:m+block_size])
                max_value = np.max(prob_grid[n:n+block_size,m:m+block_size])
                if fabs(max_value) > fabs(min_value) + num_stdev*std_value :
                    block_max = np.argmax(prob_grid[n:n+block_size,m:m+block_size])
                    n_plt = n +block_max/block_size
                    m_plt = m +block_max%block_size
                    if m_plt > n_pts/2 :                # make sure the local maximum labels fit inside image window
                        text_x = m_plt - 2
                    else:
                        text_x = m_plt+ 1
                    if n_plt > n_pts/2 :
                        text_y = n_plt -2
                    else:
                        text_y = n_plt + 4

                    pylab.annotate('('+str(m_plt)+','+str(n_plt)+')', xy=(m_plt, n_plt), xytext=(text_x,text_y))
                    pylab.plot([m_plt],[n_plt],'wo',markersize=4)
                    print "local max (x,y)=(%.2f,%.2f,%.2f)" % ((m_plt-n_pts/2)*box_size,(n_plt-n_pts/2)*box_size,max_value)

                                                        # compute angle w.r.t. x axis for this local maximum
                    r_lm = vector((m_plt-n_pts/2),(n_plt-n_pts/2))
                    if r_lm.y > 0:                      # account for whether the vector points up (above x-axis) or down
                        angle_lm = arccos(dot(norm(r_lm),(1,0,0)))
                    else:
                        angle_lm = -1.*arccos(dot(norm(r_lm),(1,0,0)))
                    angle_lm_mean += angle_lm           # compile running average of angles
                    lm_counter += 1                     # count # local maxima

                    if beamforming != 1:                # write full 2D localization data
                        ALS_write.write(str(t)+" "+str((m_plt-n_pts/2)*box_size)+" "+str((n_plt-n_pts/2)*box_size)+" "+str(max_value)+"\n") # save local maxima


    ax.set_xlim(0, n_pts)
    ax.set_ylim(0, n_pts)
    ax.set_xticks(arange(0,n_pts,block_size))
    ax.set_yticks(arange(0,n_pts,block_size))
    pylab.savefig(ALS_file_corename[0]+"_"+str(time_index)+"_.png")

    # compute & plot global maximum now:
    global_max = np.argmax(prob_grid)
    n_plt = global_max/n_pts
    m_plt = global_max%n_pts
    pylab.plot([m_plt],[n_plt],'wx',markersize=5)

    print "global maximum at [x,y]=",m_plt,n_plt,((m_plt-n_pts/2)*box_size,(n_plt-n_pts/2)*box_size)

    # *** compute and plot radial distribution at maximum angle:
    if lm_counter != 0:
        angle_lm_mean /= lm_counter
    print "mean angle = ",angle_lm_mean*360./(2.*pi)

    if beamforming == 1:                                    # write only angular data if only beamforming
        ALS_write.write(str(t)+" "+str(angle_lm_mean*360./(2.*pi))+"\n") # save time(s), angle (degrees)

    pylab.draw()                                            # replaced pylab.show() since this apparently keeps it from hanging
                                                            # see http://matplotlib.sourceforge.net/faq/howto_faq.html#use-show
                                                            # in particular, call savefig() before calling show!
    print "done with time ",t," s"
    time_index += 1                                         # keeps track of which time counter we are on

print "white dots = local maxima in regions ,",block_size*box_size," meter square"

ifile.close()                                               # done -- close sound file
ALS_write.close()                                           # and the file with local maxima and angles stored
pylab.show()                                                # show all plots finally;  will need to close these windows to close gracefully

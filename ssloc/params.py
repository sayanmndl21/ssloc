import numpy

beamforming = 0
rnorm_flag = 0
add_flag = 1
plot_local_maxima = 1
num_stdev = 2.5
num_blocks = 8
blur_image_flag = 1
gaussian_kernel = 2

v_sound = 343
f_s = 44100
dt = 1/float(f_s)
del_x = dt*v_sound
bandwidth = 5e3
signal_duration = 0.1

hop_size = 0.1
if hop_size > signal_duration:
    hopsize = signal_duration

time_array_flag = 0 #set 1 if we want to use discrete time array values

if time_array_flag == 1:
    time_array = [15.213,51.0,70.603,111.806,156.397,184.619,252.988,290.141,319.861] # times are stored in a starting array like this one
    t_i = time_array[0]                               # set initial time here from array entries
    hop_size = f_s*(time_array[1] -time_array[0])   # redefine hop_size as # samples between each time in array
else:
    t_i= 15.213                     # *** starting time
    t_f= 70.603                     # *** ending time
    t_f= t_f + hop_size
    time_array = numpy.arange(t_i,t_f,hop_size)
                                        # sets up the time to analyze array using the hop_size,start and stop times
#### set up the spatial grid that we do our computations on ##############################################################
grid_pts = 100                      # *** number of boxes across our square grid
box_size = 0.200                    # *** edge length of each grid box in meters
if v_sound/(2.*3.14159*bandwidth) <= box_size:
    print("Each grid box is ",box_size," m to an edge, which is >= ",v_sound/(2.*3.14159*bandwidth), "the smallest allowable box_size from the uncertainty principle")
else:
    print("Your grid box size was too big; it has been set to ",v_sound/(2.*3.14159*bandwidth), "the smallest allowable box_size from the uncertainty principle")
    box_size = v_sound/(2.*3.14159*bandwidth) # size of each box edge should be >= our resolution in distances from the cross-correlations
grid_size = box_size*grid_pts                 # width of field of view along x, y and z directions;  10 meter for now -- about right for the distance to our sound sources at most!

box_samples = box_size/(2.*v_sound*dt)        # an array that holds the number of samples that correspond to the travel time for each box_size/2; used to match grid boxes to del TOA's

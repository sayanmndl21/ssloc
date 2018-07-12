#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 15:02:19 2018

@author: sayan
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import os, sys
cdir = os.getcwd()
sys.path.append(cdir)
from scipy import signal
from operator import itemgetter
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter

data, fs = sf.read('900111.wav', dtype='float32')
left, right = data[0::2], data[1::2]
lf, rf = np.fft.rfft(left), np.fft.rfft(right)
"""----------------------------GOOD FILTER FOR DRONE FREQUENCY-------------------------------------"""
lowpass =4500
highpass = 50000
lf[:lowpass], rf[:lowpass] = 0, 0
lf[50:70], rf[50:70] = 0, 0
#lf[highpass:], rf[highpass:] = 0,0
nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
ns = np.column_stack((nl,nr)).ravel()
sd.play(ns, fs, blocking = True)
#############################################################
from scipy.signal import hilbert
import numpy as np
from matplotlib.pyplot import plot

sensor = data
plot(sensor)

analytical_signal = hilbert(sensor)
plot(analytical_signal.real)
plot(analytical_signal.imag)

amplitude_envelope = np.abs(analytical_signal)
plot(amplitude_envelope)
###############################################################
import matplotlib.pyplot as plt

plt.figure(1)
a = plt.subplot(211)
#r = 2**16/2
#a.set_ylim([-r, r])
a.set_xlabel('time [s]')
a.set_ylabel('sample value [-]')
x = np.arange(220500)/441000
plt.plot(x, left)
b = plt.subplot(212)
b.set_xscale('log')
b.set_xlabel('frequency [Hz]')
b.set_ylabel('|amplitude|')
plt.plot(abs(lf))
"""----------------------------GOOD FILTER FOR DRONE FREQUENCY-------------------------------------"""

def lpc(signal, order, axis=-1):
    """Compute the Linear Prediction Coefficients.
    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:
      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]
    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.
    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)
    Returns
    -------
    a : array-like
        the solution of the inversion.
    e : array-like
        the prediction error.
    k : array-like
        reflection coefficients.
    Notes
    -----
    This uses Levinson-Durbin recursion for the autocorrelation matrix
    inversion, and fft for the autocorrelation computation.
    For small order, particularly if order << signal size, direct computation
    of the autocorrelation is faster: use levinson and correlate in this case."""
    n = signal.shape[axis]
    if order > n:
        raise ValueError("Input signal must have length >= order")

    r = acorr_lpc(signal, axis)
    return levinson(r, order, axis)

def _acorr_last_axis(x, nfft, maxlag):
    a = np.real(ifft(np.abs(fft(x, n=nfft) ** 2)))
    return a[..., :maxlag+1] / x.shape[-1]

def acorr_lpc(x, axis=-1):
    """Compute autocorrelation of x along the given axis.
    This compute the biased autocorrelation estimator (divided by the size of
    input signal)
    Notes
    -----
        The reason why we do not use acorr directly is for speed issue."""
    if not np.isrealobj(x):
        raise ValueError("Complex input not supported yet")

    maxlag = x.shape[axis]
    nfft = 2 ** nextpow2(2 * maxlag - 1)

    if axis != -1:
        x = np.swapaxes(x, -1, axis)
    a = _acorr_last_axis(x, nfft, maxlag)
    if axis != -1:
        a = np.swapaxes(a, -1, axis)
    return a

def levinson(r, order, axis = -1):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.
    Parameters
    ----------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real, and corresponds
        to the autocorelation at lag 0 for linear prediction.
    order : int
        order of the recursion. For order p, you will get p+1 coefficients.
    axis : int, optional
        axis over which the algorithm is applied. -1 by default.
    Returns
    -------
    a : array-like
        the solution of the inversion (see notes).
    e : array-like
        the prediction error.
    k : array-like
        reflection coefficients.
    Notes
    -----
    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation: ::
                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :         :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
    with respect to a. Using the special symmetry in the matrix, the inversion
    can be done in O(p^2) instead of O(p^3).
    Only double argument are supported: float and long double are internally
    converted to double, and complex input are not supported at all.
    """
    if axis != -1:
        r = np.swapaxes(r, axis, -1)
    a, e, k = levinson_1d(r, order)
    if axis != -1:
        a = np.swapaxes(a, axis, -1)
        e = np.swapaxes(e, axis, -1)
        k = np.swapaxes(k, axis, -1)
    return a, e, k

def lpc_ref(signal, order):
    """Compute the Linear Prediction Coefficients.
    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:
      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]
    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.
    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)
    Notes
    ----
    This is just for reference, as it is using the direct inversion of the
    toeplitz matrix, which is really slow"""
    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size+order]
        phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi))
    else:
        return np.ones(1, dtype = signal.dtype)

def levinson_1d(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.
    Parameters
    ---------
    r : array-like
        input array to invert (since the matrix is symmetric Toeplitz, the
        corresponding pxp matrix is defined by p items only). Generally the
        autocorrelation of the signal for linear prediction coefficients
        estimation. The first item must be a non zero real.
    Notes
    ----
    This implementation is in python, hence unsuitable for any serious
    computation. Use it as educational and reference purpose only.
    Levinson is a well-known algorithm to solve the Hermitian toeplitz
    equation:
                       _          _
        -R[1] = R[0]   R[1]   ... R[p-1]    a[1]
         :      :      :          :      *  :
         :      :      :          _      *  :
        -R[p] = R[p-1] R[p-2] ... R[0]      a[p]
                       _
    with respect to a (  is the complex conjugate). Using the special symmetry
    in the matrix, the inversion can be done in O(p^2) instead of O(p^3).
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if n < 1:
        raise ValueError("Cannot operate on empty array !")
    elif order > n - 1:
        raise ValueError("Order should be <= size-1")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1/r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order+1, r.dtype)
    # temporary array
    t = np.empty(order+1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in xrange(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])

        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e, k

'''---------------------------UTILITY FUNCTIONS------------------------------'''

# Load numpy (audio) files
def load_npy(filename):
    return np.load(filename)

# Split an file into multiple parts
def split_files(filename, num):
    data = np.load(filename)
    totallength = int(len(data)/44100)
    segmentlength = int(totallength/num)
    i = 0
    while(i < num - 1):
        split = data[i*segmentlength*44100: (i+1)*segmentlength*44100]
        print(len(split))
        np.save('{}_{}'.format(filename, i), split)
        i += 1
    np.save('{}_{}'.format(filename, i), data[i*segmentlength*44100:])

# Have a generator that yield seperate files from one master file.
def split_seconds(filename, length = 44100*4, split_length = 44100):
    data = np.load(filename, 'r')
    totallength = len(data)
    i = 0
    while(i*split_length + length < totallength):
        split = np.asarray(data[i*split_length: i*split_length+length])
        i += 1
        yield split

'''---------------------------WELCH ALGORITHM--------------------------------'''

# Given audio, transform the data onto the frequency space with welch function.
def spectrum(audio, frequency = 44100, bandpass = []):
    f, psd = signal.welch(audio, frequency, nperseg = 4096, window = 'hamming')
    f, psd = filter_frequency(f, psd, bandpass, frequency)
    return f, psd

# Filter the f, psd to contain only the frequencies.
def filter_frequency(f, psd, bandpass, fs = 44100):
    if len(bandpass) == 0:
        return f, psd
    left = bandpass[0]
    right = bandpass[1]
    inlef = np.searchsorted(f, left)
    inrte = np.searchsorted(f, right, side='right')
    f = f[inlef: inrte]
    psd = psd[inlef: inrte]
    return f, psd

'''---------------------------BANDPASS FILTERING-----------------------------'''

# Generate bandpass filters
def butter_bandpass(lowcut, highcut, fs, order=9):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, bandpass, fs = 44100, order = 3):
    b, a = butter_bandpass(bandpass[0], bandpass[1], fs, order)
    data = lfilter(b, a, data)
    return data

'''---------------------------FIND MAX IN PEAK RANGE-------------------------'''

# Find local maximums in an array
def max_in_range(data, interval):
    leng = len(data)
    maxdex = set()
    for i in range(leng - interval):
        local = data[i:i+interval]
        index = np.argmax(local) + i
        maxdex.add(index)
    return maxdex

# Improved version of max_in_range
def max_range(data, interval):
    leng = len(data)
    maxdex = list()
    middle = int(interval/2)
    for i in range(leng - interval):
        # Padding increases the range as frequency increases
        padding = int(i/50)
        if i+interval+padding < leng:
            local = data[i-padding:i+interval+padding]
        else:
            break
        localmax = np.max(local)
        if discard_max(local, localmax):
            continue
        if data[i+middle] == localmax:
            index = i + middle
            maxdex.append(index)
    return maxdex

# Helper function for improved max_range
def discard_max(interval, amax, threshold = 1.5):
    minimum = np.min(interval)
    if amax/minimum < threshold:
        return True
    return False

'''------------------INDIVIDUAL FFT OF SHORTER PERIOD OF TIME----------------'''

# Old fft algorithm before using welch algorithm
def fft(nparray, fs):
    ham = np.hamming(len(nparray))
    nparray = nparray * ham
    freq = np.fft.rfft(nparray)
    freq = freq[:-1]
    nfreq = np.real(freq)
    psd = nfreq**2
    psd = welchlocal(psd)
    multitude = fs/len(nparray)
    f = (np.arange(len(nparray))+1)*multitude
    return psd

# Sooth the fft frequency using some averaging function.
def welchlocal(psd):
    leng = len(psd)
    newleng = int(leng/5)
    psd = psd[:newleng+6]
    welch = []
    i = 0
    while(i < newleng):
        if i < 5:
            welch.append(np.sum(psd[i:i+5]))
        elif i > newleng:
            welch.append(np.sum(psd[i-5:i]))
        else:
            welch.append(np.sum(psd[i-2:i+3]))
        i += 1
    return welch

'''---------------------------PEAK CALCULATIONS------------------------------'''

def peak_calc(psdarray, index, irange):
    # irange is the range of interest.
    maxpeak = 10
    left_bound = index-irange
    right_bound = index+irange
    maxi = max(psdarray[index-irange:index+irange])
    minleft = np.amin(psdarray[index-irange:index])
    while psdarray[left_bound] <= minleft:
        minleft = psdarray[left_bound]
        if left_bound < 4:
            break
        left_bound -= 1
    minright = np.amin(psdarray[index:index+irange])
    while psdarray[right_bound] <= minright:
        minright = psdarray[right_bound]
        right_bound += 1
    mincons = max(minleft, minright)
    peakstrength = psdarray[index]**2/mincons/maxi
    if peakstrength > maxpeak:
        peakstrength = maxpeak
    return peakstrength

# Calculate Peak from long averages of base rather than local spectra.
def new_peak(psdarray, index, irange):
    if index < 20:
        averages = np.average(psdarray[:40])
    else:
        averages = np.average(psdarray[index-20:index+20])
    strength = psdarray[index]/averages
    return strength

# Map
def peak_map(psdarray, irange = 4):
    peaklist = np.zeros(irange + 2)
    for i in range(irange + 2,len(psdarray)-50):
        # peaklist = np.append(peaklist,new_peak(psdarray, i, irange))
        peaklist = np.append(peaklist,peak_calc(psdarray, i, irange))
    padding = np.zeros(50)
    peaklist = np.append(peaklist, padding)
    return peaklist

# Identify the peaks of the peaklist ..
def peak_assign(peaklist):
    peaks = []
    maxpeak = 10
    for i in range(len(peaklist)):
        if i < 5:
            continue
        if i > len(peaklist) - 5:
            break
        if peaklist[i] == maxpeak:
            peaks.append(i)
            continue
        if peaklist[i] > 1 and peaklist[i] > peaklist[i-1] and peaklist[i] > peaklist[i+1]:
            peaks.append(i)
    peaks = np.asarray(peaks)
    return peaks

# Superimpose peak_calc onto analysis
def peak_impose(ax, frequency, peaklist):
    x = list(range(len(peaklist)))
    ax2 = ax.twinx()
    gf.std_graph(ax2, frequency[x], peaklist, c = 'b')
    peaksdex = peak_assign(peaklist)
    gf.button_grapher(ax2, frequency, peaksdex, peaklist)

'''------------IDENTIFICATION OF DRONES USING HARMONICS DECISION-------------'''

# Identify a drone from the result of harmonics - CORE.
def identify(result):
    av_power = np.sum(result[:,1])/result.shape[0]
    av_match = np.sum(result[:,2])/result.shape[0]
    mask = result[:, 2] > av_match
    # print('%.02f' % av_match, end = ' ')
    if np.sum(mask) < 1:
        return False
    selected = result[mask]
    mask = selected[:, 1] > av_power
    if np.sum(mask) < 1:
        return False
    selected = selected[mask]
    quants = []
    for each in selected:
        quant = (each[1]) * each[2]
        quants.append((each[0], quant, each[1], each[2]))
    strong = max(quants, key=itemgetter(1))
    return strong

def detect(strong):
    metric = strong[2] + (strong[3] - 5)*10 - 120
    if metric > 40:
        return True
    return False

# Print out identify's output
def print_identify(result):
    if not result:
        print('No drone is detected')
    else:
        if 120 < result[0] < 170:
            print('Harmonics detected at %.02f Hz, Strength %.02f' % (result[0], result[2]-120))
        else:
            print('No drone is detected')

"""-----------------------------------------------------------------------------------------------------------"""

#!/usr/bin/env python3
import numpy as np
import os, sys
sys.path.append('/Users/aperocky/workspace/Labwork/Drone_Project/Audio-detection/engines')
import graph_functions as gf
import audio_algorithms as aa
from matplotlib import pyplot as plt

# Peak calculation algorithm used to map frequency array in to 'Peakiness' array.
def peak_calcc(psdarray, index):
    # print(index)
    irange = 4 + int(index/50)
    maxi = max(psdarray[index-irange:index+irange])
    minleft = np.amin(psdarray[index-irange:index])
    minright = np.amin(psdarray[index:index+irange])
    mincons = max(minleft, minright)
    return psdarray[index]**2/mincons/maxi

def peak_calc(psdarray, index):
    # irange is the range of interest.
    irange = 6
    left_bound = index-irange
    right_bound = index+irange
    maxi = max(psdarray[index-irange:index+irange])
    minleft = np.amin(psdarray[index-irange:index])
    while psdarray[left_bound] <= minleft:
        minleft = psdarray[left_bound]
        if left_bound < 4:
            break
        left_bound -= 1
    minright = np.amin(psdarray[index:index+irange])
    while psdarray[right_bound] <= minright:
        minright = psdarray[right_bound]
        right_bound += 1
    mincons = max(minleft, minright)
    peakstrength = psdarray[index]**2/mincons/maxi
    if peakstrength > 100:
        peakstrength = 100
    return peakstrength

# Map
def peak_map(psdarray):
    irange = 8
    peaklist = np.zeros(irange + 2)
    for i in range(irange + 2,len(psdarray)-50):
        peaklist = np.append(peaklist,peak_calc(psdarray, i))
    padding = np.zeros(50)
    peaklist = np.append(peaklist, padding)
    return peaklist

# Identify the peaks of the peaklist ..
def peak_assign(peaklist):
    peaks = []
    for i in range(len(peaklist)):
        if i < 5:
            continue
        if i > len(peaklist) - 5:
            break
        if peaklist[i] == 100:
            peaks.append(i)
            continue
        if peaklist[i] > 1 and peaklist[i] > peaklist[i-1] and peaklist[i] > peaklist[i+1]:
            peaks.append(i)
    peaks = np.asarray(peaks)
    return peaks

# Superimpose peak_calc onto analysis
def peak_impose(ax, frequency, peaklist):
    x = list(range(len(peaklist)))
    ax2 = ax.twinx()
    gf.std_graph(ax2, frequency[x], peaklist, c = 'b')
    peaksdex = peak_assign(peaklist)
    gf.button_grapher(ax2, frequency, peaksdex, peaklist)

def runfromfft(f, psd):
    ax = gf.init_image()
    gf.semi_graph(ax, f, psd)
    maxdex = aa.max_range(psd, 20)
    # gf.button_grapher(ax, f, maxdex, psd)
    peaklist = aa.peak_map(psd)
    aa.peak_impose(ax, f, peaklist)
    plt.show()

if __name__ == '__main__':
    filename = sys.argv[1]
    bandpass = []
    if len(sys.argv) > 3:
        lowerbound = int(sys.argv[2])
        upperbound = int(sys.argv[3])
        bandpass = [lowerbound, upperbound]
    audio = aa.load_npy(filename)
    bandpass = [80, 10000]
    f, psd = aa.spectrum(audio, bandpass = bandpass)
    ax = gf.init_image()
    gf.semi_graph(ax, f, psd, label = filename)
    # maxdex = aa.max_range(psd, 20)
    # peaklist = aa.peak_map(psd)
    # plt.savefig('%s.png' % filename.split('.')[0])
    plt.show()

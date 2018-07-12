#this is rocky's code
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os, sys
from itertools import chain


"""----------------------------Rocky's Welch Algorithm------------------------------------------------------"""
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

"""-------------------------------------------Mapping-----------------------------------------------------------------"""

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


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os, sys
from feature_extraction import calcspectrum as cs
from itertools import chain
import operator as op

# Find the nearest frequency's index in the array
def find_nearest(frequency, values):
    index = 0
    indexes = []
    diffsave = []
    for i in range(len(frequency)):
        diff = np.abs(frequency[i] - values[index])
        diffsave.append(diff)
        if len(diffsave) < 2:
            continue
        if diffsave[-2] == min(diffsave):
            indexes.append(i-1)
            index += 1
            if index == len(values):
                break
            diffsave = list()
    return indexes

# Peak_calculation.
def peak_calc(psdarray, index):
    # print(index)
    irange = 4
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
    return psdarray[index]**2/mincons/maxi

# Map
def peak_map(psdarray):
    peaklist = np.zeros(10)
    for i in range(10,len(psdarray)-50):
        peaklist = np.append(peaklist,peak_calc(psdarray, i))
    padding = np.zeros(50)
    peaklist = np.append(peaklist, padding)
    return peaklist

# Identify the peaks of the peaklist ..
def peak_assign(peaklist):
    peaks = []
    for i in range(len(peaklist)):
        if i < 10:
            continue
        if i > len(peaklist) - 5:
            break
        if peaklist[i] > 0.8 and peaklist[i] > peaklist[i-1] and peaklist[i] > peaklist[i+1]:
            peaks.append(i)
    peaks = np.asarray(peaks)
    return peaks

# Complete the guess work and return an array of frequency checked.
def guesswork(frequency, psdarray, peaklist, peaksdex):
    guess = np.arange(135, 160, 0.25)
    output = []
    for each in guess:
        loudness = 0
        hits = 0
        hitsloc = []
        harmonics = each * np.arange(2,15)
        hardex = find_nearest(frequency, harmonics)
        for i in hardex:
            loudness += peaklist[i]*(10)
            if i in peaksdex:
                hitsloc.append(i)
                hits += 1
        output.append([loudness, hits])
    output = [list(a) for a in zip(guess, output)]
    for i in range(len(output)):
        each = output[i]
        nlist = []
        nlist.append(each[0])
        nlist.extend(each[1])
        output[i] = nlist
    output = np.array(output)
    return output

def run(audio):
    f, psd = cs.spectrum(audio)
    peaklist = cs.peak_map(psd)
    peaksdex = cs.peak_assign(peaklist)
    output = guesswork(f, psd, peaklist, peaksdex)
    return output

"""--------------------------Indentification code--------------------------------------------"""

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
    quantl = []
    metrics = []
    psds = []
    for each in selected:
        quant = (each[1]) * each[2]
        metric = each[1] + (each[2] - 5)*10 - 120
        quants.append((each[0], metric, quant, each[1], each[2]))
        metrics.append(metric)
        quantl.append(quant)
        psds.append(each[1])
    newlist = np.multiply(metric,quantl)
    idx = f(newlist, 6)
    quantn = [quants[j] for j in idx]
    metricn = [metrics[j] for j in idx]
    psdn = [psds[j] for j in idx]
    if quants:
        strong = max(quants, key = op.itemgetter(2))
        metricc = strong[1]
        if metricc > 40:
            boolian = True
        else:
            boolian = False
    else:
        boolian = False       
    return quantn, metricn, psdn, boolian

##double check for drone
def detect(quants):
    strong = max(quants, key = op.itemgetter(2))
    metric = strong[1]
    if metric > 40:
        return True
    return False

def psddetectionresults(data):
    out = run(data)
    q,m,p,b = identify(out)
    p = np.array(p)
    return p, b

def max_indices(arr, k):
    '''
    Returns the indices of the k first largest elements of arr
    (in descending order in values)
    '''
    assert k <= arr.size, 'k should be smaller or equal to the array size'
    arr_ = arr.astype(float)  # make a copy of arr
    max_idxs = []
    for _ in range(k):
        max_element = np.max(arr_)
        if np.isinf(max_element):
            break
        else:
            idx = np.where(arr_ == max_element)
        max_idxs.append(idx)
        arr_[idx] = -np.inf
    return max_idxs

def f(a,N):
    return np.argsort(a)[::-1][:N]


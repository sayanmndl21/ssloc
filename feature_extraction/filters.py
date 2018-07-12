import sounddevice as sd
import soundfile as sf
import numpy as np
import os, sys
from scipy import signal
from operator import itemgetter
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter
from scipy.signal import hilbert

def noisefilter(data, lowpass = 4500, highpass = 50000):
    left, right = data[0::2], data[1::2]
    lf, rf = np.fft.rfft(left), np.fft.rfft(right)
    lf[:lowpass], rf[:lowpass] = 0, 0
    lf[50:70], rf[50:70] = 0, 0
    lf[highpass:], rf[highpass:] = 0,0
    nl, nr = np.fft.irfft(lf), np.fft.irfft(rf)
    ns = np.column_stack((nl,nr)).ravel().astype(np.float32)
    return ns

def genhilbert(data):
    sensor = data
    analytical_signal = hilbert(sensor)
    amplitude_envelope = np.abs(analytical_signal)
    return analytical_signal, amplitude_envelope

"""-------------------------------Filters from Rocky's code---------------------------------------"""


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


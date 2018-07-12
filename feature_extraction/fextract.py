import librosa
import librosa.display
import scipy.io.wavfile as wavf
import numpy as np
import wave
import soundfile as sf
import os, sys, csv, re

def extract_feature(X, sample_rate):
    #X, sample_rate = librosa.load(file_name)

    # sftf
    stft = np.abs(librosa.stft(X))

    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def print_wave_info(file_name):
    wf = wave.open(file_name, "r")
    chan = wf.getnchannels()
    swidth = wf.getsampwidth()
    sfreq = wf.getframerate()
    nframes = wf.getnframes()
    params = wf.getparams()
    time =  float(wf.getnframes()) / wf.getframerate()
    return chan,swidth,sfreq,nframes,params,time


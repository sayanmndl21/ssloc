import pyaudio
from numpy import zeros,linspace,short,fromstring,hstack,transpose,log
from scipy import fft
from time import sleep

sensitivity = 0.1
tone = 3500
bandwidth = 30
beeplength = 8
alarmbeeps = 4
resetlength = 10
clearlength = 30
debug = False
frequencyout = False

n_sample = 2048
sr = 44100

pa = pyaudio.PyAudio()
openaudio = pa.open(format = pyaudio.paInt16, channels =1, rate = sr, input = True, frames_per_buffer =n_sample)

cblip = 0
cbeep = 0
creset = 0
cclear = 0
alarm = False

while True:
    while openaudio.get_read_available()< n_sample:
        sleep(0.01)
        audio_data = fromstring(openaudio.read(
        openaudio.get_read_available()), dtype=short)[-n_sample:]
    normalize_data = audio_data / (32*1024)
    intensity = abs(fft(normalize_data))[:n_sample/2]
    freq = linspace(0.0,float(sr)/2, num = n_sample/2)
    if frequencyout:
        which = intensity[1:].argmax()+1
        if which != len(intensity)-1:
            y0,y1,y2 = log(intensity[which-1:which+2:])
            x1 = (y2 - y0) * .5/ (2*y1 - y2 - y0)
            frq = (which+x1)*sr/n_sample
        else:
            frq = which*sr/n_sample
    if max(intensity[(freq<tone+bandwidth)&(freq>tone-bandwidth)]) > max(intensity[(freq<tone-1000)&(freq>tone-2000)])+sensitivity:
        if frequencyout:
            print("\t\t\t\tfreq=",frq)
        cblip+=1
        creset=0
        if debug: print("\t\tBlip",cblip)
        if (cblip>=beeplength):
            cblip=0
            creset=0
            cbeep+=1
            if debug: print("\tBeep",cbeep)
            if (cbeep>=alarmbeeps):
                cclear=0
                alarm=True
                print("Alarm!")
                cbeep=0
    else:
        if frequencyout:
            print("\t\t\t\tfreq=",frq)
        cblip=0
        creset+=1
        if debug: print("\t\t\treset",creset)
        if (creset>=resetlength):
            creset=0
            cbeep=0
            if alarm:
                cclear+=1
                if debug: print("\t\tclear",cclear)
                if cclear>=clearlength:
                    cclear=0
                    print("Cleared alarm!")
                    alarm=False
    sleep(0.01)

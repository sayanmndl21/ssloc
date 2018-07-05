import os, sys
import math
import numpy as np

sys.path.append("ssloc")

#import ssloc.localize as loc
from ssloc.soundrecorder import *
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



#### start program #########

sr = recorder()
sr.setup()
sr.continuerecord()
raw_data = sr.getAudio()
signal_level = round(abs(loudness(raw_data)))

print(sr.getAudio(), flush = True)

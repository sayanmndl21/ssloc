import os, sys
import math
import numpy as np

sys.path.append("ssloc")

import ssloc.localize as loc
from ssloc.soundrecorder import record
from ssloc.params import *
from ssloc.micconfig import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.optimize as opt
import signal
from scipy import signal
import unittest
from pprint import pprint
import bisect, time

keep_unittest_logs = False
unittests_bin_dir = "tests"
unittests_log_dir = "data_log"
unittests_file_pattern = "^test_[a-zA-Z0-9_]*.*$"

cmd_dict = {
    "sh": "bash"
    "py": "python"
}

class TC:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

###### Save a file of probablities vs. grid location:  (time, x,y,z, prob) #####################################################

ALS_file = os.path.abspath("logdata")             # get entire pathname of input file

ALS_file_corename=os.path.splitext(ALS_file)  # get only the core (no extension) of input file for use in naming output files

ALS_name = ALS_file_corename[0]+"_.dat"

ALS_write = open(ALS_name, 'w')
##################################################################3

#### start program #########

sr = record()

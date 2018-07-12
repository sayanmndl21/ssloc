import numpy as np
import pandas as pd


def get_parsed_mfccdata(mfcc,chroma,mel,contrast,tonnetz):
    mfcc_data = []
    features = np.empty((0,193))
    ext_features = np.hstack([mfcc,chroma,mel,contrast,tonnetz])
    features = np.vstack([features, ext_features])
    cols = ["features", "shape"]
    mfcc_data.append([features, features.shape])
    mfcc_pd = pd.DataFrame(data = mfcc_data, columns = cols)
    flat = [mfcc_pd['features'][i].ravel() for i in range(mfcc_pd.shape[0])]
    mfcc_pd['sample'] = pd.Series(flat, index = mfcc_pd.index)
    mfcc_test_data = np.array(list(mfcc_pd[:]['sample']))
    return mfcc_test_data

def get_parsed_lpcdata(lpc,rlpc,psd):
    mfcc_data = []
    features = np.empty((0,26))
    ext_features = np.hstack([lpc[1:],rlpc,psd])
    features = np.vstack([features, ext_features])
    cols = ["features", "shape"]
    mfcc_data.append([features, features.shape])
    mfcc_pd = pd.DataFrame(data = mfcc_data, columns = cols)
    flat = [mfcc_pd['features'][i].ravel() for i in range(mfcc_pd.shape[0])]
    mfcc_pd['sample'] = pd.Series(flat, index = mfcc_pd.index)
    lpc_test_data = np.array(list(mfcc_pd[:]['sample']))
    return lpc_test_data
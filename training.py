import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from matplotlib import style
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from six.moves import cPickle as pickle
from six.moves import range
from sklearn import utils
from sklearn.svm import NuSVC
from sklearn.svm import SVC
import itertools
from sklearn import model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import pickle, scipy
import csv
from sklearn.utils import shuffle

pickle_flag = 0
style.use("ggplot")
metafile = 'tenis.csv' #load dataset
data = pd.read_csv(metafile)
df = pd.DataFrame(data)
#train_X = df.iloc[:-180, 10:].values
#train_y = df.iloc[:-150, 1:3].values
#test_y = df.iloc[-150:, 1:3].values
mfcc_data = []
#lpc_data = []
with open(metafile, 'r',newline='') as f:
    reader = csv.DictReader(f, delimiter=',')
    i=0
    mfccl = []
    chromal = []
    mell = []
    spectl = []
    tonnetzl = []
    lpcl = []
    rlpcl= []
    psdl =[]
    for row in reader:
        #fileid = float(row['ID'])
        if True:#fileid > 9000000:
            mfcc = [float(t) for t in row['MFCC'].strip("[]").split()]
            chroma = [float(t) for t in row['CHROMA'].strip("[]").split()]
            mel = [float(t) for t in row['MELSPECTROGRAM'].strip("[]").split()]
            spect = [float(t) for t in row['SPECTRALCONTRAST'].strip("[]").split()]
            tonnetz = [float(t) for t in row['TONNETZ'].strip("[]").split()]
            #lpc = [float(t) for t in row['LPCCOEFF'].strip("[]").split()]
            #rlpc = [float(t) for t in row['RCOEFF'].strip("[]").split()]
            #psd = [float(t) for t in row['PSD'].strip("[,]").split(",")]
            h = [(t) for t in row['Height'].strip("[]").split()]
            d = [(t) for t in row['Distance'].strip("[]").split()]
            if str(h[0].strip("[]"))[-1:] == "m" and str(d[0].strip("[]"))[-1:] == "m":
                h = str(h[0].strip("[]"))[:-1]
                d = str(d[0].strip("[]"))[:-1]
                coord = np.array([float(h), float(d)])
            else:
                coord = np.array([float(h[0]), float(d[0])])
            v=np.linalg.norm(coord-np.array([0,0]))
        if v<=15:
            label="vnear"
            label1 = "drone"
        elif v>15 and v<=35:
            label="near"
            label1 = "drone"
        elif v>35 and v<=60:
            label1 = "drone"
            label="midrange"
        elif v>60 and v<=120:
            label1 = "drone"
            label = "far"
        elif v>120 and v<=150:
            label1 = "drone"
            label= "vfar"
        elif v > 150:
            label1 ="drone"
            label = "vfar"
        elif h[0] == "nan":
            label = "no_drone"
        features = np.empty((0,193))
        #fpfeatures = np.empty((0,26))
        ext_features = np.hstack([mfcc,chroma,mel,spect,tonnetz])
        #fpext_features = np.hstack([lpc,rlpc,psd])
        features = np.vstack([features,ext_features])
        #fpfeatures = np.vstack([fpfeatures,fpext_features])
        i+=1
        if True:#not np.isnan(lpc[0]):
            mfccl.append(mfcc)
            chromal.append(chroma)
            mell.append(mel)
            spectl.append(spect)
            tonnetzl.append(tonnetz)
            #lpcl.append(lpc)
            #psdl.append(psd)
            #rlpcl.append(rlpc)
            mfcc_data.append([features, features.shape, label])
            #lpc_data.append([fpfeatures, fpfeatures.shape, label1])
cols=["features", "shape","label"]
mfcc_pd = pd.DataFrame(data = mfcc_data, columns=cols)
mfcc_pd.sample(frac=1).reset_index(drop=True)
#lpc_pd = pd.DataFrame(data = lpc_data, columns=cols)


le = LabelEncoder()
label_num = le.fit_transform(mfcc_pd["label"])
label_num1 = le.fit_transform(lpc_pd["label"])

 # one hot encode
ohe = OneHotEncoder()
onehot = ohe.fit_transform(label_num.reshape(-1, 1))
onehot1 = ohe.fit_transform(label_num1.reshape(-1, 1))
#   for i in range(5):
#    mfcc_pd[le.classes_[i]] = onehot[:,i].toarray()
def labelling(pddata, ln):
    pddata.insert(loc=3, column='label_id', value=ln)
    labels = set(pddata['label'])
    print(labels)
    cnt = [[label,list(pddata['label']).count(label)] for label in labels]
    dict_cnt = dict(cnt)
    print(dict_cnt)
    cnt_cols=["classes","occurence"]
    count_pd = pd.DataFrame(data = cnt, columns=cnt_cols)
    return dict_cnt

dictmfcc = labelling(mfcc_pd, label_num)
dictlpc = labelling(lpc_pd, label_num1)

#see distribution
plt.figure(1)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('test2png.png', dpi=100)

fig.set_size_inches(18.5, 10.5, forward=True)
a = plt.subplot(211)
plt.bar(range(len(dictmfcc)), dictmfcc.values(), align='center')
plt.xticks(range(len(dictmfcc)), dictmfcc.keys())
plt.show()
fig.set_size_inches(18.5, 10.5, forward=True)
b = plt.subplot(212)
plt.bar(range(len(dictlpc)), dictlpc.values(), align='center')
plt.xticks(range(len(dictlpc)), dictlpc.keys())
plt.show()

it = int(len(data)*0.8)
iv = int(len(data)*0.2)
"""--------------------------------MFCC DATA--------------------------------------------"""
labels = set(mfcc_pd['label_id'])
mapping = []
for label_id in labels:
    label_name = set(mfcc_pd.loc[mfcc_pd['label_id'] == label_id]['label'])
    mapping.append((label_id,label_name))
label_mapping = dict(mapping)
label_mapping

ll = [mfcc_pd['features'][i].ravel() for i in range(mfcc_pd.shape[0])]
mfcc_pd['sample'] = pd.Series(ll, index=mfcc_pd.index)
del mfcc_pd['features']

train_data = np.array(list(mfcc_pd[:-iv]['sample']))
train_label = np.array(list(mfcc_pd[:-iv]['label_id']))
validation_data = np.array(list(mfcc_pd[-iv:]['sample']))
validation_label = np.array(list(mfcc_pd[-iv:]['label_id']))

print(train_data.shape)
print(type(train_data))
 
"""------------------------------------LPC DATA------------------------------"""

labels1 = set(lpc_pd['label_id'])
mapping = []
for label_id in labels1:
    label_name1 = set(lpc_pd.loc[lpc_pd['label_id'] == label_id]['label'])
    mapping.append((label_id,label_name1))
label_mapping1 = dict(mapping)
label_mapping1

ll1 = [lpc_pd['features'][i].ravel() for i in range(lpc_pd.shape[0])]
lpc_pd['sample'] = pd.Series(ll1, index=lpc_pd.index)
del lpc_pd['features']

lpc_train_data = np.array(list(lpc_pd[:-iv]['sample']))
lpc_train_label = np.array(list(lpc_pd[:-iv]['label_id']))
lpc_validation_data = np.array(list(lpc_pd[-iv:]['sample']))
lpc_validation_label = np.array(list(lpc_pd[-iv:]['label_id']))

print(lpc_train_data.shape)
print(type(lpc_train_data))

"""----------------------------------------------------------------------------"""
def confusion(true, predicted):
    matrix = np.zeros([6,6])
    #d = 0
    for t, p in zip(true, predicted):
        matrix[t,p] += 1.5
    #    d += 1
    #print(d)
    return matrix


svm = OneVsRestClassifier(NuSVC(nu=.02, kernel='poly', decision_function_shape='ovr'))
svm1 = OneVsRestClassifier(NuSVC(nu=.8, kernel='linear', decision_function_shape='ovr'))

svmmodel = svm.fit(train_data, train_label)

if pickle_flag == 1:
    joblib.dump(svm, 'input/detection_new.pkl')      
else:
    svc_prediction = svmmodel.predict(validation_data)
    svc_accuracy = np.sum(svc_prediction == validation_label)/validation_label.shape[0]
    print(svc_accuracy)
    classe_names = label_mapping.values()
    matrix = confusion(validation_label, svc_prediction)
    
    plt.figure(figsize=[10,10])
    plt.imshow(matrix, cmap=plt.cm.Blues, interpolation='nearest',  vmin=0, vmax=100)
    plt.colorbar()
    plt.title('SVM Confusion Map', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.grid(b=False)
    plt.yticks(range(5), classe_names, fontsize=14)
    plt.xticks(range(5), classe_names, fontsize=14, rotation='vertical')
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], '.2f'),horizontalalignment="center",color="black")
    plt.show()

in_flag = input("Want to continue? y/n ")

if in_flag == "y":
    svmmodel_lpc = svm1.fit(lpc_train_data, lpc_train_label)
    
    svc_prediction1 = svmmodel_lpc.predict(lpc_validation_data)
    svc_accuracy1 = np.sum(svc_prediction1 == lpc_validation_label)/lpc_validation_label.shape[0]
    print(svc_accuracy1)
    classe_names1 = label_mapping1.values()
    matrix1 = confusion(lpc_validation_label, svc_prediction1)
    
    plt.figure(figsize=[10,10])
    plt.imshow(matrix1, cmap=plt.cm.Blues, interpolation='nearest',  vmin=0, vmax=100)
    plt.colorbar()
    plt.title('SVM Confusion Map', fontsize=18)
    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.grid(b=False)
    plt.yticks(range(5), classe_names1, fontsize=14)
    plt.xticks(range(5), classe_names1, fontsize=14, rotation='vertical')
    for i, j in itertools.product(range(matrix1.shape[0]), range(matrix1.shape[1])):
        plt.text(j, i, format(matrix1[i, j], '.2f'),horizontalalignment="center",color="black")
    plt.show()

else:
    print("done")


#model = svm.SVC( C                        =   0.01,
#                 gamma                    =  'auto',
#                 kernel                   =  'linear',
#                 degree                   =   3,
#                 class_weight             =  'balanced',
#                 coef0                    =   0.0,
#                 decision_function_shape  =   None,
#                 probability              =   False,
#                 max_iter                 =  -1,
#                 tol                      =   0.001,
#                 cache_size               = 700,
#                 random_state             =   None,
#                 shrinking                =   True,
#                 verbose                  =   False
#                 )
#
#xlist2 = xlist[1][-180:]
#model.fit(train_x[:,1:2], train_y)
#
#lab_enc = preprocessing.LabelEncoder()
#encoded = lab_enc.fit_transform(train_y[:,1:2])
#model.fit(xlist1, encoded)

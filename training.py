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
import pickle
import csv

pickle_flag = 1
style.use("ggplot")
metafile = 'recordings.csv'
data = pd.read_csv(metafile)
df = pd.DataFrame(data)
#train_X = df.iloc[:-180, 10:].values
#train_y = df.iloc[:-150, 1:3].values
#test_y = df.iloc[-150:, 1:3].values
mfcc_data = []
with open(metafile, 'r',newline='') as f:
    reader = csv.DictReader(f, delimiter=',')
    i=0
    mfccl = []
    chromal = []
    mell = []
    spectl = []
    tonnetzl = []
    for row in reader:
        mfcc = [float(t) for t in row['MFCC'].strip("[]").split()]
        chroma = [float(t) for t in row['CHROMA'].strip("[]").split()]
        mel = [float(t) for t in row['MELSPECTROGRAM'].strip("[]").split()]
        spect = [float(t) for t in row['SPECTRALCONTRAST'].strip("[]").split()]
        tonnetz = [float(t) for t in row['TONNETZ'].strip("[]").split()]
        h = [float(t) for t in row['Height'].strip("[]").split()]
        d = [float(t) for t in row['Distance'].strip("[]").split()]
        coord = np.array([h, d])
        v=np.linalg.norm(coord-np.array([0,0]))
        if v<=15:
            label="vnear"
        elif v>15 and v<=35:
            label="near"
        elif v>35 and v<=60:
            label="midrange"
        elif v>60 and v<=100:
            label = "far"
        elif v>100:
            label= "vfar"
        features = np.empty((0,193))
        ext_features = np.hstack([mfcc,chroma,mel,spect,tonnetz])
        features = np.vstack([features,ext_features])
        i+=1
        mfccl.append(mfcc)
        chromal.append(chroma)
        mell.append(mel)
        spectl.append(spect)
        tonnetzl.append(tonnetz)
        mfcc_data.append([features, features.shape, label])
cols=["features", "shape","label"]
mfcc_pd = pd.DataFrame(data = mfcc_data, columns=cols)

le = LabelEncoder()
label_num = le.fit_transform(mfcc_pd["label"])
 # one hot encode
ohe = OneHotEncoder()
onehot = ohe.fit_transform(label_num.reshape(-1, 1))
#   for i in range(5):
#    mfcc_pd[le.classes_[i]] = onehot[:,i].toarray()

mfcc_pd.insert(loc=3, column='label_id', value=label_num)
labels = set(mfcc_pd['label'])
print(labels)
cnt = [[label,list(mfcc_pd['label']).count(label)] for label in labels]
dict_cnt = dict(cnt)
print(dict_cnt)
cnt_cols=["classes","occurence"]
count_pd = pd.DataFrame(data = cnt, columns=cnt_cols)

#see distribution
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig('test2png.png', dpi=100)

fig.set_size_inches(18.5, 10.5, forward=True)
plt.bar(range(len(dict_cnt)), dict_cnt.values(), align='center')
plt.xticks(range(len(dict_cnt)), dict_cnt.keys())
plt.show()

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

train_data = np.array(list(mfcc_pd[:]['sample']))
train_label = np.array(list(mfcc_pd[:]['label_id']))
validation_data = np.array(list(mfcc_pd[:350]['sample']))
validation_label = np.array(list(mfcc_pd[:350]['label_id']))

print(train_data.shape)
print(type(train_data))

def confusion(true, predicted):
    matrix = np.zeros([5,5])
    #d = 0
    for t, p in zip(true, predicted):
        matrix[t,p] += 1.5
    #    d += 1
    #print(d)
    return matrix

svm = OneVsRestClassifier(NuSVC(nu=.02, kernel='poly', decision_function_shape='ovr'))
svmmodel = svm.fit(train_data, train_label)

if pickle_flag == 1:
    joblib.dump(svm, 'input/detection.pkl')      
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

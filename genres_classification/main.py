# Beat tracking example
from __future__ import print_function
import matplotlib.pyplot as plt
import librosa
from dtw import dtw
import matplotlib.pyplot as plt
import numpy as np
import librosa
from sklearn import svm
import IPython.display
from IPython.display import Image
from os import walk
from os.path import isfile, join
from sklearn.externals import joblib


def predict_fun(name_file, clf):
    predict_y, predict_sr = librosa.load(name_file)
    predict_mfcc = librosa.feature.mfcc(y=predict_y, sr=predict_sr)
    mfcc_predict = []
    for item in predict_mfcc:
        mfcc_predict.extend(item)
    print(str(name_file) + str(clf.predict(mfcc_predict[:25000])))


def train_model():
    path_to_folder = "./genres_train"

    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    array_of_path = []
    for (dirpath, dirnames, filenames) in walk(path_to_folder):
        array_of_path.extend([join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")])

    array_y = []
    array_sr = []

    for path in array_of_path:
        y, sr = librosa.load(path=path)
        array_y.extend([y])
        array_sr.extend([sr])

    array_mfcc = []

    librosa.feature.mfcc(array_y[0], array_sr[0])

    for i in range(0,len(array_y)):
        array_mfcc.append(librosa.feature.mfcc(y=array_y[i], sr=array_sr[i]))


    array_of_array_mfcc = [[]]

    for i in range(0, len(array_mfcc)):
        mfcc_feat = []
        for frame in array_mfcc[i]:
            mfcc_feat.extend(frame)
        array_of_array_mfcc.append(mfcc_feat)


    print(array_of_array_mfcc)

    # X = [mfcc_feat[:100], mfcc_feat2[:100], mfcc_feat3[:100], mfcc_feat4[:100]]
    array_of_labels = []
    for (dirpath, dirnames, filenames) in walk(path_to_folder):
        array_of_labels.extend([filename.split("/")[-2] for filename in [join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")]])


    X = [x[0:25000] for x in array_of_array_mfcc if len(x) > 0]
    Y = [[x] for x in array_of_labels]


    clf = svm.SVC(probability=True)
    clf.fit(X, Y)

    return clf


# MAIN PART
choose = int(input("Do you want to train new model[1] or use old one[0]"))
if choose == 1:
    clf = train_model()
    joblib.dump(clf, 'train_model.pkl')
else:
    try:
        clf = joblib.load('train_model.pkl')
    except FileNotFoundError:
        clf = train_model()


predict_fun("./genres/blues/blues.00020.au", clf)
predict_fun("./genres/disco/disco.00020.au", clf)
predict_fun("./genres/hiphop/hiphop.00020.au", clf)
predict_fun("./genres/jazz/jazz.00020.au", clf)
predict_fun("./genres/rock/rock.00020.au", clf)
predict_fun("./genres/metal/metal.00020.au", clf)


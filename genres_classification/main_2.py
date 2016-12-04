# Beat tracking example
import librosa
from sklearn import svm
from os import walk
from os.path import isfile, join
from sklearn.externals import joblib

path_to_folder = "./genres_train"

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)


def predict_fun(name_file, clf):
    predict_y, predict_sr = librosa.load(name_file)
    predict_mfcc = librosa.feature.mfcc(y=predict_y, sr=predict_sr)
    mfcc_predict = []
    for item in predict_mfcc:
        mfcc_predict.extend(item)
    # print(str(name_file) + str(clf.decision_function(mfcc_predict[:25000])))
    print(str(name_file) + str(clf.predict(mfcc_predict[:25000])))


def train_model():

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

    for i in range(0,len(array_y)):
        array_mfcc.append(librosa.feature.mfcc(y=array_y[i], sr=array_sr[i]))

    array_of_array_mfcc = [[]]

    for i in range(0, len(array_mfcc)):
        mfcc_feat = [0.0] * len(max(array_mfcc[i], key=len))
        for frame in array_mfcc[i]:
            for x in range(0, len(mfcc_feat[x])):
                mfcc_feat[x] = mfcc_feat[x] + frame[x]

        # for x in range(0, array_mfcc[i]):
        #     mfcc_feat[x] /= float(len(frame))

        array_of_array_mfcc.append(mfcc_feat)


    print(array_of_array_mfcc)

    # X = [mfcc_feat[:100], mfcc_feat2[:100], mfcc_feat3[:100], mfcc_feat4[:100]]
    array_of_labels = []
    for (dirpath, dirnames, filenames) in walk(path_to_folder):
        array_of_labels.extend([filename.split("/")[-2] for filename in [join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")]])


    X = [x[0:25000] for x in array_of_array_mfcc if len(x) > 0]
    Y = [[x] for x in array_of_labels]


    clf = svm.SVC()
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

import os
array_of_labels = dirlist = [ item for item in os.listdir(path_to_folder) if os.path.isdir(os.path.join(path_to_folder, item)) ]
print(array_of_labels)

print(clf)

predict_fun("./genres/blues/blues.00020.au", clf)
predict_fun("./genres/disco/disco.00020.au", clf)
predict_fun("./genres/hiphop/hiphop.00020.au", clf)
predict_fun("./genres/jazz/jazz.00020.au", clf)
predict_fun("./genres/rock/rock.00020.au", clf)
predict_fun("./genres/metal/metal.00020.au", clf)



# from scikits.talkbox.features import mfcc
# http://pydub.com/ for slice audio
import librosa
from sklearn import svm
from os import walk
from os.path import isfile, join
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import copy

path_to_folder = "./genres_train"
glob_n_mfcc = 20
glob_sr = 22050
glob_duration = 30.0
glob_n_vectors = 13

def flaten_matrix(matrix):
    concat = []
    for row in matrix:
        concat.extend(row)
    return concat

def extract_genre(filenames, dirpath):
    return [filename.split("/")[-2] for filename in [join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")]]

def preprocess_mfcc(mfcc):
    mfcc_cp = copy.deepcopy(mfcc)
    for i in range(mfcc.shape[1]):
        mfcc_cp[:,i] = mfcc[:,i] - np.mean(mfcc[:,i])
        mfcc_cp[:,i] = mfcc_cp[:,i]/np.max(np.abs(mfcc_cp[:,i]))
    return mfcc_cp


def predict_genre(name_file, clf):
    predict_y, predict_sr = librosa.load(name_file, sr=glob_sr, duration=glob_duration)
    current_features = librosa.feature.mfcc(y=predict_y, sr=predict_sr, n_mfcc=glob_n_mfcc)[:glob_n_vectors]
    return clf.predict(np.reshape(current_features, (1, -1)))


def train_model():

    # Form paths for files from directory path_to_folder

    paths = []
    for (dirpath, dirnames, filenames) in walk(path_to_folder):
        paths.extend([join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")])

    # Load the audio as a waveform `y`
    # Store the sampling rate as `sr`

    array_y = []
    array_sr = []

    for path in paths:
        y, sr = librosa.load(path=path, sr=glob_sr, duration=glob_duration)
        array_y.append(y)
        array_sr.append(sr)


    # Extract features using MFCC

    features_mfcc = []
    for i in range(len(array_y)):
        current_features = librosa.feature.mfcc(y=array_y[i], sr=array_sr[i], n_mfcc=glob_n_mfcc)[:glob_n_vectors] # take first 13 vectors
        features_mfcc.append(flaten_matrix(current_features)) # add obtained vectors in row

    # Extract label for each file

    labels = []
    for (dirpath, dirnames, filenames) in walk(path_to_folder):
        labels.extend(extract_genre(filenames, dirpath))

    # Create a classification model using SVM

    clf = svm.SVC()

    # scores = cross_val_score(clf,features_mfcc_cut , labels, cv=5)
    # print("Scores:\n", scores)

    clf.fit(features_mfcc, labels)

    return clf


def main():
    # choose = int(input("Do you want to train new model[1] or use old one[0]"))
    # if choose == 1:
    #     clf = train_model()
    #     joblib.dump(clf, 'train_model.pkl')
    # else:
    #     try:
    #         clf = joblib.load('train_model.pkl')
    #     except FileNotFoundError:
    #         clf = train_model()

    clf = train_model()

    print("MANUAL TESTING:")


    print("./genres/blues/blues.00020.au", predict_genre("./genres/blues/blues.00020.au", clf))
    print("./genres/disco/disco.00020.au", predict_genre("./genres/disco/disco.00020.au", clf))
    print("./genres/hiphop/hiphop.00020.au", predict_genre("./genres/hiphop/hiphop.00020.au", clf))
    print("./genres/jazz/jazz.00020.au", predict_genre("./genres/jazz/jazz.00020.au", clf))
    print("./genres/rock/rock.00020.au", predict_genre("./genres/rock/rock.00020.au", clf))
    print("./genres/metal/metal.00020.au", predict_genre("./genres/metal/metal.00020.au", clf))

    # print("./genres/blues/blues.00020.au", predict_genre("./genres/blues/blues.00020.au", clf, min_length))
    # print("./genres/disco/disco.00020.au", predict_genre("./genres/disco/disco.00020.au", clf, min_length))
    # print("./genres/hiphop/hiphop.00020.au", predict_genre("./genres/hiphop/hiphop.00020.au", clf, min_length))
    # print("./genres/jazz/jazz.00020.au", predict_genre("./genres/jazz/jazz.00020.au", clf, min_length))
    # print("./genres/rock/rock.00020.au", predict_genre("./genres/rock/rock.00020.au", clf, min_length))
    # print("./genres/metal/metal.00020.au", predict_genre("./genres/metal/metal.00020.au", clf, min_length))

    # print("./genres/blues/blues.00021.au", predict_genre("./genres/blues/blues.00021.au", clf, min_length))
    # print("./genres/disco/disco.00021.au", predict_genre("./genres/disco/disco.00021.au", clf, min_length))
    # print("./genres/hiphop/hiphop.00021.au", predict_genre("./genres/hiphop/hiphop.00021.au", clf, min_length))
    # print("./genres/jazz/jazz.00021.au", predict_genre("./genres/jazz/jazz.00021.au", clf, min_length))
    # print("./genres/rock/rock.00021.au", predict_genre("./genres/rock/rock.00021.au", clf, min_length))
    # print("./genres/metal/metal.00021.au", predict_genre("./genres/metal/metal.00021.au", clf, min_length))
    #
    # print("./genres/blues/blues.00022.au", predict_genre("./genres/blues/blues.00022.au", clf, min_length))
    # print("./genres/disco/disco.00022.au", predict_genre("./genres/disco/disco.00022.au", clf, min_length))
    # print("./genres/hiphop/hiphop.00022.au", predict_genre("./genres/hiphop/hiphop.00022.au", clf, min_length))
    # print("./genres/jazz/jazz.00022.au", predict_genre("./genres/jazz/jazz.00022.au", clf, min_length))
    # print("./genres/rock/rock.00022.au", predict_genre("./genres/rock/rock.00022.au", clf, min_length))
    # print("./genres/metal/metal.00022.au", predict_genre("./genres/metal/metal.00022.au", clf, min_length))


if __name__ == "__main__":
    main()

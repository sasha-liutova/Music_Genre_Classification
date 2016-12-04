
# from scikits.talkbox.features import mfcc
import librosa
from sklearn import svm
from os import walk
from os.path import isfile, join
from sklearn.externals import joblib
import numpy as np

path_to_folder = "./genres_train"

def extract_genre(filenames, dirpath):
    return [filename.split("/")[-2] for filename in [join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")]]

def predict_genre(name_file, clf):
    predict_y, predict_sr = librosa.load(name_file)
    predict_mfcc = librosa.feature.mfcc(y=predict_y, sr=predict_sr)
    mfcc_predict = []
    for item in predict_mfcc:
        mfcc_predict.extend(item)
    return clf.predict(np.reshape(mfcc_predict[:25000], (1, -1)))


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
        y, sr = librosa.load(path=path)
        array_y.append(y)
        array_sr.append(sr)


    # Extract features using MFCC

    features_mfcc = []
    for i in range(0,len(array_y)):
        current_features = librosa.feature.mfcc(y=array_y[i], sr=array_sr[i])
        features_flat = []
        for frame in current_features:
            features_flat.extend(frame)
        features_mfcc.append(features_flat)

    # Extract label for each file

    labels = []
    for (dirpath, dirnames, filenames) in walk(path_to_folder):
        labels.extend(extract_genre(filenames, dirpath))

    X = [x[0:25000] for x in features_mfcc]

    # Create a classification model using SVM

    clf = svm.SVC()
    clf.fit(X, labels)

    return clf


def main():
    choose = int(input("Do you want to train new model[1] or use old one[0]"))
    if choose == 1:
        clf = train_model()
        joblib.dump(clf, 'train_model.pkl')
    else:
        try:
            clf = joblib.load('train_model.pkl')
        except FileNotFoundError:
            clf = train_model()

    print("./genres/blues/blues.00020.au", predict_genre("./genres/blues/blues.00020.au", clf))
    print("./genres/disco/disco.00020.au", predict_genre("./genres/disco/disco.00020.au", clf))
    print("./genres/hiphop/hiphop.00020.au", predict_genre("./genres/hiphop/hiphop.00020.au", clf))
    print("./genres/jazz/jazz.00020.au", predict_genre("./genres/jazz/jazz.00020.au", clf))
    print("./genres/rock/rock.00020.au", predict_genre("./genres/rock/rock.00020.au", clf))
    print("./genres/metal/metal.00020.au", predict_genre("./genres/metal/metal.00020.au", clf))


if __name__ == "__main__":
    main()

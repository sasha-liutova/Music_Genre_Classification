
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

test_path = './genres_train'
glob_path_train = './genres_split/train'
glob_path_test = './genres_split/test'
glob_n_mfcc = 20
glob_sr = 22050
glob_duration = 30.0
glob_n_vectors = 13

def flaten_matrix(matrix):
    """
    :param matrix: any 2D matrix
    :return: vector, consisting of concatenated rows from original matrix
    """
    concat = []
    for row in matrix:
        concat.extend(row)
    return concat


def paths_from_folder(folder):
    """
    :param folder: string path to folder with files
    :return: paths to all files in folder
    """
    paths = []
    for (dirpath, dirnames, filenames) in walk(folder):
        paths.extend([join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")])
    return paths

def extract_genre(filenames, dirpath):
    """
    :param filenames:
    :param dirpath:
    :return: list of genres of files
    """
    return [filename.split("/")[-2] for filename in [join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")]]

def extract_genres_from_paths(files):
    """
    :param files: list of paths to files
    :return: list of genres of files
    """
    genres=[]
    for file in files:
        genres.append(file.split('/')[-1].split('.')[0])
    return genres

# def preprocess_mfcc(mfcc):
#     mfcc_cp = copy.deepcopy(mfcc)
#     for i in range(mfcc.shape[1]):
#         mfcc_cp[:,i] = mfcc[:,i] - np.mean(mfcc[:,i])
#         mfcc_cp[:,i] = mfcc_cp[:,i]/np.max(np.abs(mfcc_cp[:,i]))
#     return mfcc_cp


def predict_genre(name_file, clf):
    """
    :param name_file:
    :param clf: model
    :return: predicted genre (string)
    """
    predict_y, predict_sr = librosa.load(name_file, sr=glob_sr, duration=glob_duration)
    current_features = librosa.feature.mfcc(y=predict_y, sr=predict_sr, n_mfcc=glob_n_mfcc)[:glob_n_vectors]
    return clf.predict(np.reshape(current_features, (1, -1)))


def train_model(folder):

    # Form paths for files from directory path_to_folder
    paths = paths_from_folder(folder)

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
    for (dirpath, dirnames, filenames) in walk(folder):
        labels.extend(extract_genre(filenames, dirpath))

    # Create a classification model using SVM

    clf = svm.SVC()

    # scores = cross_val_score(clf,features_mfcc_cut , labels, cv=5)
    # print("Scores:\n", scores)
    for vector, file in zip(features_mfcc, paths):
        print(file, len(vector))

    clf.fit(features_mfcc, labels)

    return clf

def test_model(clf, folder):
    paths = paths_from_folder(folder)
    expected_labels = extract_genres_from_paths(paths)
    predicted_labels = []
    for file in paths:
        predicted_labels.append(predict_genre(file, clf))
    correct, false = 0, 0
    for predicted,expected in zip(predicted_labels, expected_labels):
        if(predicted == expected):
            correct += 1
        else:
            false += 1
        print(expected, predicted)
    accuracy = correct/(correct+false)
    print('accuracy: ', accuracy)

def main():
    choose = int(input("Do you want to train new model[1] or use old one[0]"))
    if choose == 1:
        clf = train_model(glob_path_train)
        joblib.dump(clf, 'train_model.pkl')
    else:
        try:
            clf = joblib.load('train_model.pkl')
        except FileNotFoundError:
            clf = train_model(glob_path_train)

    test_model(clf, glob_path_test)


if __name__ == "__main__":
    main()

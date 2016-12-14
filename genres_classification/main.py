
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
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


test_path = './genres_train'
glob_path_train = './genres_split/train'
glob_path_test = './genres_split/test'
glob_n_mfcc = 13
glob_sr = 22050
glob_duration = 30.0

def flaten_matrix(matrix):
    """
    :param matrix: any 2D matrix
    :return: vector, consisting of concatenated rows from original matrix
    """
    concat = []
    for row in matrix:
        concat.extend(row)
    return concat

def kl(p, q):
	"""Kullback-Leibler divergence D(P || Q) for discrete distributions

    The Kullback–Leibler divergence is an asymmetric information theoretic divergence measure. It is a measure of difference
    between two probability distributions.
	Parameters

	The Kullback-Leibler divergence measures the distance between two distributions: P(X,Y)P(X,Y) and P(X)⋅P(Y)
	----------
	p, q : array-like, dtype=float, shape=n
	Discrete probability distributions.
	"""
	p = np.asarray(p, dtype=np.float)
	q = np.asarray(q, dtype=np.float)

	return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def extract_min_max_vectors(matrix):
    pass

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


def extract_features(paths):
    # Load the audio as a waveform `y`
    # Store the sampling rate as `sr`

    features_mfcc = []

    choose = int(input("Do you want to use new features[1] or use old one[0]"))

    if choose == 1:

        for path in paths:
            offset = 0
            duration_song = librosa.get_duration(filename=path)
            if duration_song > glob_duration:
                offset = duration_song/2 - glob_duration/2 # around middle of song
            y, sr = librosa.load(path=path, sr=glob_sr, offset= offset, duration=glob_duration)
            # print(path)

            # Extract features using MFCC

            #S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=65, fmax = 8000)
            #mfcc = librosa.feature.mfcc(S=librosa.logamplitude(S))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=glob_n_mfcc)
            #mfcc = librosa.feature.melspectrogram(y=y, sr=sr)

            X = []
            for i in range(len(mfcc)):
                mfcc_len = len(mfcc[i])
                X.append(np.mean(mfcc[i][int(mfcc_len / 10):int(mfcc_len * 9 / 10)], axis=0))


            # Extract tempo

            hop_length = 512
            oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
            tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length = hop_length)
            # Compute global onset autocorrelation
            ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
            ac_global = librosa.util.normalize(ac_global)
            # Estimate the global tempo for display purposes
            tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=hop_length)

            # print(tempo)
            X.append(tempo)

            #zero crossing rate
            zero_crossing = librosa.feature.zero_crossing_rate(y)

            # print(np.mean(zero_crossing))
            X.append(np.mean(zero_crossing))

            features_mfcc.append(X)  # add obtained vectors in row

            # print("\n")

        joblib.dump(features_mfcc, 'features_mfcc.data')
    else:
        try:
            features_mfcc = joblib.load('features_mfcc.data')
        except FileNotFoundError:
            print("File not found!")
            features_mfcc = extract_features(glob_path_train)

    return features_mfcc

def predict_genre(name_file, clf):
    """
    :param name_file:
    :param clf: model
    :return: predicted array of probabilities of genres
    """

    features = extract_features([name_file])[0]

    return clf.predict_proba(np.reshape(features, (1, -1)))


def predict_genre_classic(name_file, clf):
    """
    :param name_file:
    :param clf: model
    :return: predicted genre (string)
    """

    features = extract_features([name_file])[0]

    return clf.predict(np.reshape(features, (1, -1)))

def train_model(folder):

    # Form paths for files from directory path_to_folder
    paths = paths_from_folder(folder)

    features = extract_features(paths)

    # Extract label for each file

    labels = []
    for (dirpath, dirnames, filenames) in walk(folder):
        labels.extend(extract_genre(filenames, dirpath))

    # Create a classification model using kNN

    clf = KNeighborsClassifier(n_neighbors=7, metric="minkowski", p=3)
    # clf = RandomForestClassifier(n_estimators=5, max_depth=None,min_samples_split=2)
    #clf = KNeighborsClassifier(n_neighbors=5, algorithm="ball_tree", n_jobs=4)

    # scores = cross_val_score(clf,features_mfcc_cut , labels, cv=5)
    # print("Scores:\n", scores)
    for vector, file in zip(features, paths): # REMOVE THIS !!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(file, len(vector))

    clf.fit(features, labels)

    return clf

def test_model(clf, folder):
    paths = paths_from_folder(folder)
    expected_labels = extract_genres_from_paths(paths)
    predicted_labels = []
    for file in paths:
        predicted_labels.append(predict_genre_classic(file, clf))
    correct, false = 0, 0
    for predicted,expected in zip(predicted_labels, expected_labels):
        if(predicted == expected):
            correct += 1
        else:
            false += 1
        print(expected, predicted)
    accuracy = correct/(correct+false)
    print('accuracy: ', accuracy)

def plot_data(labels, features):

    # calculating how many unique clusters there are
    unique_clusters = []
    for l in labels:
        if not l in unique_clusters:
            unique_clusters.append(l)
    n_labels = len(unique_clusters)

    # creating structure for displaying, containing pairs of coordinates and cluster numbers
    plot_data = []
    for label, point in zip(labels, features):
        index = unique_clusters.index(label)
        plot_data.append([point, index])

    # list of colors for visualization
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
              '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    if n_labels > len(colors):
        color = iter(plt.cm.rainbow(np.linspace(0, 1, n_labels)))
        colors = []
        for i in range(n_labels):
            c = next(color)
            colors.append(c)

    # displaying plot
    for element in plot_data:
        point, label = element[0], element[1]
        plt.scatter(point[0], point[1], color=colors[label], s=5)
    plt.show()

def visualize_data(folder):

    try:
        features = pickle.load(open("features.py", "rb"))
    except FileNotFoundError:
        # Form paths for files from directory path_to_folder
        paths = paths_from_folder(folder)

        features = extract_features(paths)

        pickle.dump(features, open("features.py", "wb"))

    tsne = TSNE(n_components=2, random_state=0)
    transformed_features = tsne.fit_transform(features)

    # Extract label for each file

    labels = []
    for (dirpath, dirnames, filenames) in walk(folder):
        labels.extend(extract_genre(filenames, dirpath))

    plot_data(labels, transformed_features)

def main():
    choose = int(input("Do you want to train new model[1] or use old one[0]"))
    if choose == 1:
        clf = train_model(glob_path_train)
        joblib.dump(clf, 'train_model_minkowski3_new_features.pkl')
    else:
        try:
            clf = joblib.load('train_model_minkowski3_new_features.pkl')
        except FileNotFoundError:
            clf = train_model(glob_path_train)

    test_model(clf, glob_path_test)

    # visualize_data(glob_path_train)


if __name__ == "__main__":
    main()

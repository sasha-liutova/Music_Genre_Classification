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
    The Kullback-Leibler divergence measures the distance between two distributions: P(X,Y)P(X,Y) and P(X)P(Y)
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.

    """

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


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
    return [filename.split("/")[-2] for filename in
            [join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")]]


def extract_genres_from_paths(files):
    """
    :param files: list of paths to files
    :return: list of genres of files
    """
    genres = []
    for file in files:
        genres.append(file.split('/')[-1].split('.')[0])
    return genres


# def preprocess_mfcc(mfcc):
#     mfcc_cp = copy.deepcopy(mfcc)
#     for i in range(mfcc.shape[1]):
#         mfcc_cp[:,i] = mfcc[:,i] - np.mean(mfcc[:,i])
#         mfcc_cp[:,i] = mfcc_cp[:,i]/np.max(np.abs(mfcc_cp[:,i]))
#     return mfcc_cp


def extract_features(paths, choose=0):
    # Extract and save features from all audios given by paths

    features_mfcc = []

    if choose == 1:

        for path in paths:
            X = extract_feature(path)
            features_mfcc.append(X)  # add obtained vector to the row

        joblib.dump(features_mfcc, 'features_mfcc.data')
    else:
        try:
            features_mfcc = joblib.load('features_mfcc.data')
        except FileNotFoundError:
            print("File not found!")
            features_mfcc = extract_features(paths, choose=1)

    return features_mfcc


def extract_feature(path):
    # Extract feature exactly from one audio, which is in path

    offset = 0
    duration_song = librosa.get_duration(filename=path)
    if duration_song > glob_duration:
        offset = duration_song / 2 - glob_duration / 2  # around middle of song
    y, sr = librosa.load(path=path, sr=glob_sr, offset=offset, duration=glob_duration)

    # Extract features using MFCC

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=glob_n_mfcc)

    X = []
    for i in range(len(mfcc)):
        mfcc_len = len(mfcc[i])
        X.append(np.mean(mfcc[i][int(mfcc_len / 10):int(mfcc_len * 9 / 10)], axis=0))

    # Extract tempo

    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Estimate the global tempo for display purposes
    tempo = librosa.beat.estimate_tempo(oenv, sr=sr, hop_length=hop_length)

    # print(tempo)
    X.append(tempo)

    # zero crossing rate
    zero_crossing = librosa.feature.zero_crossing_rate(y)

    # print(np.mean(zero_crossing))
    X.append(np.mean(zero_crossing))

    return X


def predict_genre_probabilities(name_file, clf):
    """
    :param name_file:
    :param clf: model
    :return: predicted array of probabilities of genres
    """

    features = [extract_feature(name_file)]
    return clf.predict_proba(np.reshape(features, (1, -1)))


def predict_genre(name_file, clf):  # DELETEEEEEEEEEEE!!!!!!!!!!!!
    """
    :param name_file:
    :param clf: model
    :return: predicted genre (string)
    """

    features = [extract_feature(name_file)]
    return clf.predict(np.reshape(features, (1, -1)))


def train_model(folder):
    # Form paths for files from directory path_to_folder
    paths = paths_from_folder(folder)
    choose = int(input("Do you want to use new features[1] or use old one[0]:"))
    features = extract_features(paths, choose=choose)

    # Extract label for each file
    labels = []
    for (dirpath, dirnames, filenames) in walk(folder):
        labels.extend(extract_genre(filenames, dirpath))


        # ONLY FOR TESTING RIGHT PARAMS FOR VALIDATOR
        # tuned_parameters = [
        #     {
        #         'n_neighbors': [1,3,5,7,9,11,13,15],
        #         'weights': ["uniform", "distance"],
        #         "metric" : ["euclidean","manhattan","chebyshev"]
        #     },
        #     {
        #         'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
        #         'weights': ["uniform", "distance"],
        #         "algorithm" : ["ball_tree", "kd_tree" , "brute"],
        #         "leaf_size": [10,20,30,40,50],
        #         "metric": ["euclidean", "manhattan", "chebyshev"]
        #     },
        #     {
        #         'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
        #         'weights': ["uniform", "distance"],
        #         "algorithm": ["ball_tree", "kd_tree", "brute"],
        #         "leaf_size": [10, 20, 30, 40, 50],
        #         "metric": ["minkowski"],
        #         "p": [1,2,3,4,5,6,7]
        #     }
        # ]

        # tuned_parameters = [
        # {
        #     "C": [0.1, 1, 4, 10, 50],
        #     "kernel": ["linear"]
        # },
        # {
        #     "C": [0.1, 0.5, 1, 2, 4, 8, 10, 100],
        #     "kernel": ["poly"],
        #     "degree": [2, 3, 4, 5],
        #     "coef0": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2, 4, 8, 16],
        #     "probability": [True],
        # },
        # {
        #     "C": [0.1, 0.5, 1, 2, 4, 8, 10, 100],
        #     "kernel": ["sigmoid"],
        #     "probability": [True],
        #
        # },
    #     {
    #         "C": [0.1, 0.5, 1, 2, 4, 8, 10, 100],
    #         "kernel": ["rbf"],
    #         "probability": [True],
    #     }
    # ]


    # clf = KNeighborsClassifier(n_neighbors=7, metric="chebyshev")

    # clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters)
    # clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=3, n_jobs=4)


    # clf = svm.SVC(C=4, kernel="linear")

    # clf.fit(X_train, y_train)
    # clf = RandomForestClassifier(n_estimators=5, max_depth=None,min_samples_split=2)
    clf = KNeighborsClassifier(n_neighbors=7, n_jobs=4)

    clf.fit(features, labels)

    return clf


def test_model(clf, folder):
    """
    tests model on predifined testing set
    :param clf: model
    :param folder: path to files from testing set
    :return: prints accuracy
    """
    paths = paths_from_folder(folder)
    expected_labels = extract_genres_from_paths(paths)
    predicted_labels = []
    for file in paths:
        predicted_labels.append(predict_genre(file, clf))
    correct, false = 0, 0
    for predicted, expected in zip(predicted_labels, expected_labels):
        if (predicted == expected):
            correct += 1
        else:
            false += 1
        print(expected, predicted)
    accuracy = correct / (correct + false)
    print('accuracy: ', accuracy)


def plot_data(labels, features):
    """
    visualizes extracted features in 2d dimension
    :param labels:
    :param features:
    :return: shows plot
    """
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
        features = pickle.load(open("features_mfcc.data", "rb"))
    except FileNotFoundError:
        # Form paths for files from directory path_to_folder
        paths = paths_from_folder(folder)

        features = extract_features(paths, choose=1)

        pickle.dump(features, open("features_mfcc.data", "wb"))

    tsne = TSNE(n_components=2, random_state=0)
    transformed_features = tsne.fit_transform(features)

    # Extract label for each file

    labels = []
    for (dirpath, dirnames, filenames) in walk(folder):
        labels.extend(extract_genre(filenames, dirpath))

    plot_data(labels, transformed_features)


def main():
    choose = int(input("Do you want to train new model[1] or use old one[0]:"))
    if choose == 1:
        clf = train_model(glob_path_train)
        joblib.dump(clf, 'model.pkl')
    else:
        try:
            clf = joblib.load('model.pkl')
        except FileNotFoundError:
            clf = train_model(glob_path_train)

    # Only for testing best validator
    # print(clf.best_estimator_)
    # print(clf.best_params_)
    # print(clf.best_score_)

    test_model(clf, glob_path_test)

    visualize_data(glob_path_train)


if __name__ == "__main__":
    main()

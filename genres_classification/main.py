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

y, sr = librosa.load("./genres/rock/rock.00000.au")
# y2, sr2 = librosa.load("./genres/blues/blues.00000.au")
# y3, sr3 = librosa.load("./genres/classical/classical.00000.au")
# y4, sr4 = librosa.load("./genres/country/country.00000.au")

array_mfcc = []

librosa.feature.mfcc(array_y[0], array_sr[0])

for i in range(0,len(array_y)):
    array_mfcc.append(librosa.feature.mfcc(y=array_y[i], sr=array_sr[i]))

mfcc =  librosa.feature.mfcc(y=y, sr=sr)
# mfcc2 =  librosa.feature.mfcc(y=y2, sr=sr2)
# mfcc3 =  librosa.feature.mfcc(y=y3, sr=sr3)
# mfcc4 =  librosa.feature.mfcc(y=y4, sr=sr4)

array_of_array_mfcc = [[]]

for i in range(0, len(array_mfcc)):
    mfcc_feat = []
    for frame in array_mfcc[i]:
        mfcc_feat.extend(frame)
    array_of_array_mfcc.append(mfcc_feat)


mfcc_feat = []
for item in mfcc:
    mfcc_feat.extend(item)
#
# mfcc_feat2 = []
# for item in mfcc2:
#     mfcc_feat2.extend(item)
#
# mfcc_feat3 = []
# for item in mfcc3:
#     mfcc_feat3.extend(item)
#
# mfcc_feat4 = []
# for item in mfcc4:
#     mfcc_feat4.extend(item)

print(array_of_array_mfcc)

# X = [mfcc_feat[:100], mfcc_feat2[:100], mfcc_feat3[:100], mfcc_feat4[:100]]
array_of_labels = []
for (dirpath, dirnames, filenames) in walk(path_to_folder):
    array_of_labels.extend([filename.split("/")[-2] for filename in [join(dirpath, f) for f in filenames if isfile(join(dirpath, f)) and f.endswith(".au")]])


X = [x[0:1000] for x in array_of_array_mfcc if len(x) > 0]
Y = [[x] for x in array_of_labels]


clf = svm.SVC()
clf.fit(X, Y)

predict_y, predict_sr = librosa.load("./genres/blues/blues.00020.au")
predict_y, predict_sr = librosa.load("./genres/disco/disco.00020.au")
predict_y, predict_sr = librosa.load("./genres/hiphop/hiphop.00020.au")
predict_y, predict_sr = librosa.load("./genres/jazz/jazz.00020.au")
predict_y, predict_sr = librosa.load("./genres/rock/rock.00020.au")
predict_y, predict_sr = librosa.load("./genres/metal/metal.00020.au")
predict_mfcc =  librosa.feature.mfcc(y=predict_y, sr=predict_sr)

mfcc_predict = []
for item in predict_mfcc:
    mfcc_predict.extend(item)

print(clf.predict(mfcc_predict[:1000]))

# ---------------------------
# # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
# chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#
# # print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
#
# # beat_times = librosa.frames_to_time(beat_frames, sr=sr)
# print(chroma_stft)
# print(mfcc)
#
# spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
# S, phase = librosa.magphase(librosa.stft(y=y))
#
# plt.plot(y)
# IPython.display.Audio(data=y, rate=sr)
# librosa.display.specshow(mfcc)
#
# # print('Saving output to beat_times.csv')
# # librosa.output.times_csv('beat_times.csv', beat_times)
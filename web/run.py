from flask import Flask, flash, render_template, request, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename

# for feature extraction and prediction
from sklearn.neighbors import KNeighborsClassifier
import librosa
import numpy as np
from sklearn.externals import joblib

# We cannot import pyhon file from different folder by default
# This is run-time way to do so
import sys
sys.path.append('./../genres_classification')
# Importing main project from folder genres_classification
import main


UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = set(['au','mp3'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret123' # have to be to secure redirect

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('Soubor neni vybrán.')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('Soubor neni vybrán.')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Nepodporovaný typ souboru. Podporovány jsou mp3 a au.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('show_result', name = filename))

    return render_template('index.html')

@app.route('/result/<name>')
def show_result(name):

    # Model.pkl have to be a type of scikit-learn classificator with method predict_proba
    clf = joblib.load('model.pkl')
    result = main.predict_genre_probabilities(name_file = name, clf = clf)
    labels = ["Blues", "Classical", "Country", "Disco", "Hip hop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    values = [int(x*10) for x in result[0]]
    colors = [ '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
              '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'  ]

    return render_template('result.html', filename= name, genre = labels[result.argmax()], values=values, labels=labels, colors=colors)

if __name__ == '__main__':
    app.run(debug=True)

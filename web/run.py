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
# Importin main project from folder genres_classification
import main


UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = set(['au'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret123'

@app.route("/image", methods=['GET'])
def login():
    location = request.args.get('pic')
    return render_template('index.html', my_string=location, 
        title="Index") 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Not supported type of file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('show_result', name = filename))
            #return redirect(url_for('uploaded_file', filename=filename))
            

    return render_template('index.html')

@app.route('/result/<name>')
def show_result(name):

    clf = joblib.load('train_model_5-knn_44.pkl')
    result = main.predict_genre(name_file = name, clf = clf)
    labels = ["Blues", "Classical", "Country", "Disco", "Hip hop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    values = [int(x*10) for x in result[0]]
    colors = [ '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
              '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'  ]

    return render_template('result.html', filename = labels[result.argmax()], values=values, labels=labels, colors=colors)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(debug=True)

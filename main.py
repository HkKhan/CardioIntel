
from flask import Flask, render_template, request
from werkzeug import secure_filename
import keras
import tensorflow as tf
import keras.layers
from keras.models import model_from_json
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import scipy.io.wavfile


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['wav'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
def buildModel():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    return model

def extractSin():
    userUpload = [f for f in listdir("uploads") if isfile(join("uploads", f))]
    filename = "uploads/" + userUpload[0]
    print("file", filename)
    fs, data = scipy.io.wavfile.read(filename)
    print("data", len(data))
    data = data.reshape(-1,1).T
    data = keras.preprocessing.keras_preprocessing.sequence.pad_sequences(data, maxlen=396900, dtype = np.float64, padding = 'post', value = 0)
    data = np.sin(data)
    data = data.reshape(1,396900,1)
    os.remove(filename)
    return data

def makeStringPrediction(prediction):
    stringPrediction = ""
    if prediction[0][3] > .45:
        return "Normal"
    if prediction[0][2] > .2:
        return "Murmur"
    if prediction[0][0] > .05:
        return "Artifact"
    if prediction[0][1] > .1:
        return "Extrasystole"

#0 is Artifact
#1 is Extrasystole
#2 is Murmur
#3 is Normal
###############################################################################
@app.route('/')
def home():
    return render_template('index0.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
      f = request.files['file']
      if f and allowed_file(f.filename):
          f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
          model = buildModel()
          prediction = model.predict(extractSin())
          stringPrediction = makeStringPrediction(prediction)
          del model
          return render_template('output.html', value=stringPrediction, artifactChance=round(100*prediction[0][0], 2), extraChance=round(100*prediction[0][1],2), murmurChance=round(100*prediction[0][2], 2), normalChance=round(100*prediction[0][3], 2))
      else:
          return render_template('wrongfile.html')

if __name__ == '__main__':
   app.run(debug = True)

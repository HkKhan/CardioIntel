
from flask import Flask, render_template, request
from werkzeug import secure_filename
import keras
import tensorflow as tf
import keras.layers
from keras.models import model_from_json
import numpy as np
import os
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
    filename = "uploads/artifact__201105061143.wav"
    print("file", filename)
    fs, data = scipy.io.wavfile.read(filename)
    print("data", len(data))
    data = data.reshape(-1,1)
    data = keras.preprocessing.keras_preprocessing.sequence.pad_sequences(data, maxlen=396900, dtype = np.float64, padding = 'post', value = 0)
    data = np.sin(data)
    return data

#0 is Artifact
#1 is Extrasystole
#2 is Murmur
#3 is Normal

@app.route('/')
def home():
    print(extractSin())
    return render_template('index0.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
      f = request.files['file']
      if f and allowed_file(f.filename):
          f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
          model = buildModel()
          #prediction = model.predict(extractSin())
          print(prediction)
          stringPrediction = "Normal"
          return render_template('output.html', value=stringPrediction)
      else:
          return render_template('wrongfile.html')

if __name__ == '__main__':
   app.run(debug = True)

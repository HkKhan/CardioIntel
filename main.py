
from flask import Flask, render_template, request
from werkzeug import secure_filename
import keras
import tensorflow as tf
import keras.layers
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['wav'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#0 is Artifact
#1 is Extrasystole
#2 is Extrasystole
#3 is Murmur
#4 is Normal

@app.route('/')
def home():
   return render_template('index0.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
      f = request.files['file']
      if f and allowed_file(f.filename):
          f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
          model = keras.models.load_model('finalModel.h5')
          #prediction = model.predict("/uploads/*.wav")
          #print(prediction)
          stringPrediction = "Normal"
          return render_template('output.html', value=stringPrediction)
      else:
          return render_template('wrongfile.html')

if __name__ == '__main__':
   app.run(debug = True)

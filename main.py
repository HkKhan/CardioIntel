
from flask import Flask, render_template, request
from werkzeug import secure_filename
from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import os

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
    return model

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
          #model = buildModel()
          #prediction = model.predict("/uploads/*.wav")
          #print(prediction)
          stringPrediction = "Normal"
          return render_template('output.html', value=stringPrediction)
      else:
          return render_template('wrongfile.html')

if __name__ == '__main__':
   app.run(debug = True)

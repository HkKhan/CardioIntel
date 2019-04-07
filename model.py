import os
import numpy as np
import scipy.io.wavfile
import keras
import keras.layers as layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import decomposition
import scipy.fftpack as fft
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
from keras.models import load_model
import random
import numba



base_dir = os.path.join('C:\\', 'Users', 'Haneef Khan', 'Downloads', 'heartbeat-sounds')

csv_file = os.path.join(base_dir, 'set_a.csv')

raw_csv_data = open(csv_file, 'rt')
csv_data = np.loadtxt(raw_csv_data, usecols = 0, skiprows = 1, delimiter = ",", dtype = np.str)

csv_file2 = os.path.join(base_dir, 'set_b.csv')

raw_csv_data2 = open(csv_file2, 'rt')
csv_data2 = np.loadtxt(raw_csv_data2, usecols = 0, skiprows = 1, delimiter = ",", dtype = np.str)

all_csv_data=np.concatenate((csv_data,csv_data2))
all_csv_data = all_csv_data.reshape(-1,1)

ohe = OneHotEncoder(categories = 'auto')
csv_data = ohe.fit_transform(csv_data.reshape(-1,1)).toarray().astype(np.float)
csv_data2 = ohe.fit_transform(csv_data2.reshape(-1,1)).toarray().astype(np.float)

set_a =[]
set_b =[]

set_a_values=np.ones((1,396900))
set_b_values=np.ones((1,396900))

path_a = os.path.join(base_dir, 'set_a')

set_a.append(os.listdir(path_a))


path_b = os.path.join('C:\\', 'Users', 'Haneef Khan', 'Downloads', 'heartbeat-sounds', 'set_b')

set_b.append(os.listdir(path_b))


for file in set_a[0]:
    filename = os.path.join(path_a,file)
    fs, data = scipy.io.wavfile.read(filename)
    data = data.reshape((-1, 1)).T
    data = keras.preprocessing.keras_preprocessing.sequence.pad_sequences(data, maxlen=396900, dtype = type(data[0][0]), padding ='post', value =0)
    set_a_values = np.vstack((set_a_values, data))
    del data
    print('set_a', set_a_values.shape)

for file in set_b[0]:
    filename = os.path.join(path_b,file)
    fs, data = scipy.io.wavfile.read(filename)
    data = data.reshape((-1, 1)).T
    data = keras.preprocessing.keras_preprocessing.sequence.pad_sequences(data, maxlen=396900, dtype = type(data[0][0]), padding ='post', value =0)
    set_b_values = np.vstack((set_b_values, data))
    del data
    print('set_b', set_b_values.shape)

set_a_values = np.sin(set_a_values)
set_b_values = np.sin(set_b_values)

set_b_values = set_b_values[1:]
set_a_values = set_a_values[1:124]


all_csv_data=np.concatenate((csv_data,csv_data2))
all_csv_data = all_csv_data.reshape(-1,1)

data = np.vstack((set_a_values,set_b_values))

ohe = OneHotEncoder(categories = 'auto')
all_csv_data = ohe.fit_transform(all_csv_data.reshape(-1,1)).toarray().astype(np.float)





pca = decomposition.PCA(n_components=1)
pca.fit_transform(data)

all_data = np.concatenate((data, all_csv_data), axis=1)
np.random.shuffle(all_data)









all_data = all_data.reshape(584,396904,1)

train_data= all_data[0:467,0:396900]
train_labels=all_data[0:467, 396900:396904,0]
test_data = all_data[467:584,0:396900]
test_labels = all_data[467:584,396900:396904,0]



cnn = keras.models.Sequential()


cnn.add(layers.Conv1D(2, kernel_size=(1), strides=(1),
                 activation='relu',
                 input_shape=(396900,1)))
cnn.add(layers.MaxPooling1D(pool_size=(2), strides=(2)))
cnn.add(layers.Conv1D(8, (1), activation='relu'))
cnn.add(layers.MaxPooling1D(pool_size=(2)))

cnn.add(layers.CuDNNLSTM(12, input_shape=(396900,1)))
#cnn.add(layers.CuDNNGRU(12))
cnn.add(layers.Dense(4,activation='softmax'))
cnn.compile(optimizer= 'sgd', loss='categorical_crossentropy',metrics=['acc'])
history = cnn.fit(train_data,train_labels,
                    epochs=5,
                    batch_size=4,
                    validation_data=(test_data,test_labels ))


#results = models.evaluate(test_data, test_targets)
history_dict= history.history

epochs = range(1,2+1)

train_loss = history_dict['loss']
test_loss = history_dict['val_loss']

plt.plot(epochs, train_loss, 'ro', label='Training Loss')
plt.plot(epochs, test_loss, 'bo', label='Validation Loss')
plt.title('Training (red) and validation loss (blue)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model_json = cnn.to_json()
with open("model.json", "w", ) as json_file:
    json_file.write(model_json)
cnn.save_weights("model.h5")

import os
import numpy as np
import scipy.io.wavfile
import keras
import keras.layers as layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from scipy.fftpack import fft
from google.colab import drive

drive.mount('/content/drive')
csv_file = '/content/drive/My Drive/HackTJ/heartbeat-sounds/set_a.csv'

raw_csv_data = open(csv_file, 'rt')
csv_data = np.loadtxt(raw_csv_data, usecols = 0, skiprows = 1, delimiter = ",", dtype = np.str)


ohe = OneHotEncoder(categories = 'auto')
csv_data = ohe.fit_transform(csv_data.reshape(-1,1)).toarray().astype(np.float)

set_a =[]
#set_b =[]

set_a_values=np.ones((1,396900))
#set_b_values=np.ones((1,396900))

path_a = '/content/drive/My Drive/HackTJ/heartbeat-sounds/set_a'

set_a.append(os.listdir(path_a))


#path_b = os.path.join('C:\\', 'Users', 'Haneef Khan', 'Downloads', 'heartbeat-sounds', 'set_b')

#set_b.append(os.listdir(path_b))


for file in set_a[0]:
    filename = os.path.join(path_a,file)
    fs, data = scipy.io.wavfile.read(filename)
    data = data.reshape((-1, 1)).T
    data = keras.preprocessing.keras_preprocessing.sequence.pad_sequences(data, maxlen=396900, dtype = type(data[0][0]), padding ='post', value =0)
    set_a_values = np.vstack((set_a_values, data))
    del data
    print('set_a', set_a_values.shape)
set_a_values = set_a_values[1:124]
set_a_values = fft(set_a_values)







set_a_values = set_a_values[0:124]
print(len(set_a_values), len(csv_data))
all_data = np.concatenate((set_a_values, csv_data), axis=1)
np.random.shuffle(all_data)

all_data = all_data.reshape(123,396904,1)

train_data= all_data[0:90,0:396900]
train_labels=all_data[0:90, 396900:396904,0]
test_data = all_data[90:123,0:396900]
test_labels = all_data[90:123,396900:396904,0]

''''
model = keras.models.Sequential()
model.add(layers.LSTM(1, input_shape=(396900,1), return_sequences = True, kernel_regularizer = keras.regularizers.l1_l2(l1=0.01, l2=0.0001), recurrent_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation = 'softmax')
          '''

cnn = keras.models.Sequential()
cnn.add(layers.Conv1D(1, kernel_size=(5), strides=(1),
                 activation='relu',
                 input_shape=(396900,1)))
cnn.add(layers.MaxPooling1D(pool_size=(2), strides=(2)))
cnn.add(layers.Conv1D(1, (5), activation='relu'))
cnn.add(layers.MaxPooling1D(pool_size=(2)))

cnn.add(layers.LSTM(10,input_shape=(99222,1)))
cnn.add(layers.Dense(4, activation='softmax'))




cnn.compile(optimizer= 'nadam', loss='categorical_crossentropy',metrics=['acc'])


history = cnn.fit(train_data,train_labels,
                    epochs=10,
                    batch_size=8,
                    validation_data=(test_data,test_labels),
                   )

#results = models.evaluate(test_data, test_targets)
history_dict= history.history

epochs = range(1,1000+1)

loss = history_dict['loss']
val_loss = history_dict['val_loss']

train_loss = loss
test_loss = val_loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks([x for x in range(10)])
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

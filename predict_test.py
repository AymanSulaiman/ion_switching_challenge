import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import seaborn as sns


import pywt
import scipy
from scipy import signal
from scipy.signal import butter, deconvolve

test_path = os.path.join('data','test.csv')
train_path = os.path.join('data','train.csv')

test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')

time_train = train_df.time
signal_train_denoised = denoise_signal(train_df.signal)
open_channels_train = train_df.open_channels

data = {
    'time': time_train,
    'signal_denoised': signal_train_denoised,
    'open_channels_train': open_channels_train
       }

train_data = pd.DataFrame(data=data)

# def deep_learning_model(X, y, inputs=2, epochs=10, outputs=11):
#     import tensorflow as tf
#     import datetime
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import MinMaxScaler, LabelEncoder
#     from tensorflow.keras.losses import sparse_categorical_crossentropy
#     from tensorflow.keras.utils import to_categorical

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     X_scaler = MinMaxScaler().fit(X_train)
#     X_train_scaled = X_scaler.transform(X_train)
#     X_test_scaled = X_scaler.transform(X_test)

#     label_encoder = LabelEncoder()
#     label_encoder.fit(y_train)
#     encoded_y_train = label_encoder.transform(y_train)
#     encoded_y_test = label_encoder.transform(y_test)

#     y_train_categorical = to_categorical(encoded_y_train)
#     y_test_categorical = to_categorical(encoded_y_test)
    
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense
    
#     model = Sequential()
#     model.add(Dense(units=100, activation='relu', input_dim=inputs))
#     model.add(Dense(units=200, activation='relu'))
#     model.add(Dense(units=300, activation='relu'))
#     model.add(Dense(units=300, activation='relu'))
#     model.add(Dense(units=300, activation='relu'))
#     model.add(Dense(units=300, activation='relu'))
#     model.add(Dense(units=300, activation='relu'))
#     model.add(Dense(units=300, activation='relu'))

#     model.add(Dense(units=100, activation='relu'))
#     model.add(Dense(units=outputs, activation='softmax'))
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     print(model.summary())
    
#     model.fit(
#         X_train_scaled,
#         y_train_categorical,
#         epochs=epochs,
#         shuffle=True,
#         verbose=2
#     )
    
    
#     print('\n----------------------------------------------------------------\n')
    
#     model_loss, model_accuracy = model.evaluate(
#     X_test_scaled, y_test_categorical, verbose=2)
#     print(f"Normal Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")
    
#     encoded_predictions = model.predict_classes(X_test_scaled[:100])
#     prediction_labels = label_encoder.inverse_transform(encoded_predictions)

#     return print(f'''
#     Predicted classes: {list([i for i in prediction_labels])}
#     Actual Labels: {[i[0] for i in y_test[:100]]}
#     ''')

X = train_data[['time','signal_denoised']]
y = train_data.open_channels_train.values.reshape(-1,1)
# q = deep_learning_model(X=X,y=y ,inputs=2,epochs = 10)
# print(q)


import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_scaler = MinMaxScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y_train = label_encoder.transform(y_train)
encoded_y_test = label_encoder.transform(y_test)

y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=2))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=300, activation='relu'))
model.add(Dense(units=200, activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=11, activation='softmax'))
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
print(model.summary())

model.fit(
    X_train_scaled,
    y_train_categorical,
    epochs=10,
    shuffle=True,
    verbose=2
)

print('\n----------------------------------------------------------------\n')

model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test_categorical, verbose=2)
print(f"Normal Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")

encoded_predictions = model.predict_classes(X_test_scaled[:100])
prediction_labels = label_encoder.inverse_transform(encoded_predictions)

print(f'''
Predicted classes: {list([i for i in prediction_labels])}
Actual Labels: {[i[0] for i in y_test[:100]]}
''')

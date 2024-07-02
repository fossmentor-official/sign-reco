import keras
from sklearn.model_selection import train_test_split
import os, random
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Activation, Input
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam
import tensorflow as tf
from tensorflow.keras.backend import mean, std
from PIL import Image
import imageio.v2 as imageio

ROWS = 190
COLS = 160
CHANNELS = 3
#TRAIN_DIR = '/Users/imac/PycharmProjects/Signature-recognition/data/train/'
#TEST_DIR = '/Users/imac/PycharmProjects/Signature-recognition/data/test/'

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the images directory
TRAIN_DIR = os.path.join(script_dir, 'data/train/')  # Update this path
TEST_DIR = os.path.join(script_dir, 'data/test/')  # Update this path

# Check if the images directory exists and contains images
if not os.path.exists("/Users/imac/PycharmProjects/pythonProject/sign-reco/data/train/"):
    raise ValueError("The images directory does not exist. Please ensure the directory is correct.")

if not os.listdir(TEST_DIR):
    raise ValueError("No images found. Please ensure the directories contain the images.")



SIGNATURE_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'K', 'L', 'M', 'N', 'O', 'P']

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(mean(tf.square(y_pred - y_true), axis=-1))

def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR + '{}'.format(fish)
    if not os.path.exists(fish_dir):
        print(f"Directory does not exist: {fish_dir}")
        return []
    images = [fish + '/' + im for im in os.listdir(fish_dir)]
    return images

def read_image(src):
    filepath = src
    im = imageio.imread(filepath)
    im_resized = np.array(Image.fromarray(im).resize((COLS, ROWS), Image.Resampling.LANCZOS))
    return im_resized

files = []
y_all = []

for fish in SIGNATURE_CLASSES:
    fish_files = get_images(fish)
    if not fish_files:
        continue
    files.extend(fish_files)
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))

if not files:
    raise ValueError("No images found. Please ensure the directories contain the images.")

y_all = np.array(y_all)
print(len(files))
print(len(y_all))

X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files):
    X_all[i] = read_image(TRAIN_DIR + im)
    if i % 1000 == 0:
        print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)

y_all = LabelEncoder().fit_transform(y_all)
y_all = to_categorical(y_all)

# Remove stratify parameter since stratification is not feasible with very few samples per class
X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.33, random_state=23)

optimizer = RMSprop(learning_rate=1e-4)
objective = 'categorical_crossentropy'

def center_normalize(x):
    return (x - mean(x)) / std(x)

print('1')
model = Sequential()
model.add(Input(shape=(ROWS, COLS, CHANNELS)))
model.add(Activation(center_normalize))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (2, 2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(len(SIGNATURE_CLASSES)))
model.add(Activation('sigmoid'))

adam = Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss=root_mean_squared_error)

early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

model.fit(X_train, y_train, batch_size=64, epochs=3, validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])

preds = model.predict(X_valid, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))

test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(test_files):
    test[i] = read_image(TEST_DIR + im)

test_preds = model.predict(test, verbose=1)
submission = pd.DataFrame(test_preds, columns=SIGNATURE_CLASSES)
submission.insert(0, 'image', test_files)
submission.head()

#submission.to_csv('/Users/imac/PycharmProjects/Signature-recognition/signatureResults.csv', index=False)
submission.to_csv('signatureResults.csv', index=False)

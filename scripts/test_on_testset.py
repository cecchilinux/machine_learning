import argparse
import sys
import os
import warnings

from keras.layers import Dense, Dropout, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import my_mod_load as load
import numpy as np
import pickle
import settings

parser = argparse.ArgumentParser(description="Traffic signs classifier")
parser.add_argument("-m", "--model", help="path to the already trained model", default="")
args = parser.parse_args()
model = args.model
dataset_gtsrb = "online"

LR = 0.01
BATCH_SIZE = 128


# -----------------------------
# load dataset manipulated
# -----------------------------

for root, dirs, files in os.walk(settings.MANIPULATED_DIR):
    for dirname in sorted(dirs, reverse=True):
        if dataset_gtsrb in dirname: # controllo tra 'online' o 'pickle'
            test_path = os.path.join(settings.MANIPULATED_DIR, dirname, "test.p")
        break
    break

#-------------------------------------------------
# Load the Dataset from pickle (.p files)
#--------------------------------------------------

warnings.filterwarnings('ignore')

with open(test_path, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']

n_test = len(X_test) # Number of testing examples.

json_file = open("{}.json".format(model), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("{}.h5".format(model))
print("Loaded model from disk\n")
print("Testing: ")
sgd = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)

loaded_model.compile(optimizer=sgd,
                metrics=['accuracy'],
                loss='categorical_crossentropy')


predicted_proba = loaded_model.predict(X_test)

labels_wild = y_test

well_aimed = 0
num_images = 0
for true_label,row in zip(labels_wild, predicted_proba):
    num_images += 1
    topk = np.argsort(row)[::-1][:1]
    topp = np.sort(row)[::-1][:1]
    if(true_label == topk[0]):
        well_aimed += 1

print("well-aimed: {}/{}".format(well_aimed, num_images))
print("well-aimed: {:.4f}".format(well_aimed/num_images))

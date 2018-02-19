import sys
import os
import warnings
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time
import re

import my_mod_logs as log
import my_mod_nets as nets

import settings

parser = argparse.ArgumentParser(description="Traffic signs classifier")
#parser.add_argument('problem', help='The problem name (inside ./in folder)')
parser.add_argument("-a", "--augmentation", help="Using augment data or not", action="store_true")
parser.add_argument("-b", "--blur", help="apply the blur function (augment data)", action="store_true")
parser.add_argument("-s", "--batch_size", help="", default='128')
parser.add_argument("-d", "--dropout", help="", default='.3')
parser.add_argument("-e", "--epochs", help="number of epochs", default='20')
parser.add_argument("-l", "--learning_rate", help="", default='1e-2')
parser.add_argument("-n", "--net", help="The net you wanna use (now locked sol178ML)", default="sol178ML")
parser.add_argument("-m", "--model", help="path to the already trained model", default="")
parser.add_argument("--dataset", help="(online, pickle)", default="online")
# parser.add_argument("-t", "--test", help="test on new images", action="store_true")
parser.add_argument("--debug", help="Print debug messages", action="store_true")
parser.add_argument("--quiet", help="Print only the evaluation", action="store_true")
args = parser.parse_args()
# problem_name = args.problem
net_name = args.net
EPOCHS = int(args.epochs)
LR = float(args.learning_rate)
BATCH_SIZE = int(args.batch_size)
dropout = float(args.dropout)
dataset_gtsrb = args.dataset
model = args.model
IMG_SIZE = 32
NUM_CLASSES = 43

eval_only = False if model == "" else True
# test_new_images = True if args.test else False
debug = True if args.debug else False
quiet = True if args.quiet else False
augmentation = True if args.augmentation else False
blur = True if args.blur else False

# #178 ML
net_name = "sol178ML"
features = [108, 108]
dense_hidden_units = [100]
dropouts = [0.2, 0.2, 0.5]

#178 ML
# net_name = "sol200ML"
# features = [108, 200]
# dense_hidden_units = [100]
# dropouts = [0.2, 0.2, 0.5]

#26
# features = [38, 64]
# dense_hidden_units = [256]
# dropouts = [0.2, 0.2, 0.5]


print("net {}".format(net_name))
print("epochs: {}".format(EPOCHS))
print("learning rate: {}".format(LR))
print("bath size: {}".format(BATCH_SIZE))
print("dropout: {}".format(dropout))


if not eval_only:
    # new model folder
    date = time.strftime("%Y-%m-%d_%H%M")
    new_model_folder = "{}".format(date)
    newpath = os.path.join(settings.MODELS_DIR, new_model_folder)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    model_name = 'model_{}-{}-{}_ep{}'.format(features[0], features[1], dense_hidden_units[0], EPOCHS)
    model_path = os.path.join(newpath, model_name)

else:
    model_path = model


# -----------------------------
# create and load the log file
# -----------------------------
log_file_name = "{}_{}_{}_{}_{}_{}_{}".format(time.strftime("%Y-%m-%d_%H%M"),
    net_name, EPOCHS, LR, BATCH_SIZE, dropout, dataset_gtsrb)

log.setup_file_logger('{}{}.log'.format(settings.LOG_DIR, log_file_name))

# -----------------------------
# load dataset manipulated
# -----------------------------

for root, dirs, files in os.walk(settings.MANIPULATED_DIR):
    for dirname in sorted(dirs, reverse=True):
        log.log("dataset used = {}".format(dirname), False)
        if dataset_gtsrb in dirname: # controllo tra 'online' o 'pickle'
            train_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train.p")
            if augmentation:
                train_aug1_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_aug1.p")
                train_aug2_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_aug2.p")
                train_aug3_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_aug3.p")
                train_aug4_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_aug4.p")
            if blur:
                train_br1_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_br1.p")

            valid_path = os.path.join(settings.MANIPULATED_DIR, dirname, "valid.p")
            test_path = os.path.join(settings.MANIPULATED_DIR, dirname, "test.p")
        break
    break

#-------------------------------------------------
# Load the Dataset from pickle (.p files)
#--------------------------------------------------

warnings.filterwarnings('ignore')

with open(train_path, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

if augmentation:
    with open(train_aug1_path, mode='rb') as f:
        train_aug1 = pickle.load(f)
    with open(train_aug2_path, mode='rb') as f:
        train_aug2 = pickle.load(f)
    with open(train_aug3_path, mode='rb') as f:
        train_aug3 = pickle.load(f)
    with open(train_aug4_path, mode='rb') as f:
        train_aug4 = pickle.load(f)

    X_train_aug, y_train_aug = train_aug1['features'], train_aug1['labels']
    X_train_aug2, y_train_aug2 = train_aug2['features'], train_aug2['labels']
    X_train_aug3, y_train_aug3 = train_aug3['features'], train_aug3['labels']
    X_train_aug4, y_train_aug4 = train_aug4['features'], train_aug4['labels']

    X_train = np.concatenate((X_train, X_train_aug, X_train_aug2, X_train_aug3, X_train_aug4), axis=0)
    y_train = np.concatenate((y_train, y_train_aug, y_train_aug2, y_train_aug3, y_train_aug4), axis=0)

if blur:
    with open(train_br1_path, mode='rb') as f:
        train_br1 = pickle.load(f)

    X_train_br, y_train_br = train_br1['features'], train_br1['labels']

    X_train = np.concatenate((X_train, X_train_br), axis=0)
    y_train = np.concatenate((y_train, y_train_br), axis=0)

with open(valid_path, mode='rb') as f:
    valid = pickle.load(f)
X_valid, y_valid = valid['features'], valid['labels']
with open(test_path, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']

n_train = len(X_train) # Number of training examples
n_test = len(X_test) # Number of testing examples.
n_valid = len(X_valid) # Number of testing examples.
n_classes = len(np.unique(y_train)) # How many unique classes/labels there are in the dataset.

log.log("Number of training examples = {}".format(n_train), False)
log.log("Number of validation examples = {}".format(n_valid), False)
log.log("Number of testing examples = {}".format(n_test) , False)
log.log("Number of classes = {}\n".format(n_classes), False)

# -----------------------------
# Step   : training
# -----------------------------


from keras.layers import Dense, Dropout, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD

from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, ModelCheckpoint



y_train = np_utils.to_categorical(y_train, 43)
y_valid = np_utils.to_categorical(y_valid, 43)
y_test = np_utils.to_categorical(y_test, 43)

# ----------------

# input image
inputs = Input(shape=(32, 32, 1))

# ---
# Stage 1
# --------
# First conv: 5x5 kernel, 1x1 stride, valid padding, outputs 28x28x108
first_layer = Convolution2D(nb_filter = features[0], nb_row = 5, nb_col = 5, border_mode='valid', subsample=(1, 1), activation='relu')(inputs)
# Max pooling: 2x2 stride, outputs 14x14x108
first_p_layer = MaxPooling2D(pool_size=(2, 2))(first_layer)
# Dropout: 0.2
drop_1 = Dropout(dropouts[0])(first_p_layer)

# ---
# Stage 2
# ----------
# Branch 1:
# Max pooling: 2x2 stride, outputs 7x7x108
second_p_layer = MaxPooling2D(pool_size=(2, 2))(drop_1)
first_input_layer = Flatten()(second_p_layer)
# Branch 2:
# Second conv: 5x5 kernel, 1x1 stride, valid padding, outputs 10x10x108
second_layer = Convolution2D(nb_filter = features[1], nb_row = 5, nb_col = 5, border_mode='valid', subsample=(1, 1), activation='relu')(drop_1)
# Max pooling: 2x2 stride, outputs 5x5x108
third_p_layer = MaxPooling2D(pool_size=(2, 2))(second_layer)
# Dropout: 0.2
drop_2 = Dropout(dropouts[1])(third_p_layer)
second_input_layer = Flatten()(drop_2)

# ---
# Classifier
# ---------
# Merge the two branches
input_layer = merge([first_input_layer, second_input_layer], mode='concat', concat_axis=1)
# Fully connected layer: 100 neurons
hidden_layer = Dense(dense_hidden_units[0], activation='sigmoid')(input_layer)
# Dropout: 0.5
drop = Dropout(dropouts[2])(hidden_layer)
# Softmax: 43 neurons
predictions = Dense(43, activation='softmax')(drop)
model = Model(input=inputs, output=predictions)



# inputs = Input(shape=(32, 32, 1))
#
# first_layer = Convolution2D(features[0], 3, 3, activation='relu')(inputs)
# first_layer = Convolution2D(features[0], 3, 3, activation='relu')(first_layer)
#
# first_p_layer = MaxPooling2D(pool_size=(2, 2))(first_layer)
# drop_1 = Dropout(dropouts[0])(first_p_layer)
#
# second_p_layer = MaxPooling2D(pool_size=(2, 2))(drop_1)
#
# first_input_layer = Flatten()(second_p_layer)
#
# second_layer = Convolution2D(features[1], 3, 3, activation='relu')(drop_1)
# second_layer = Convolution2D(features[1], 3, 3, activation='relu')(second_layer)
#
# third_p_layer = MaxPooling2D(pool_size=(2, 2))(second_layer)
# drop_2 = Dropout(dropouts[1])(third_p_layer)
#
# second_input_layer = Flatten()(drop_2)
#
# input_layer = merge([first_input_layer, second_input_layer], mode='concat', concat_axis=1)
# hidden_layer = Dense(dense_hidden_units[0], activation='sigmoid')(input_layer)
# drop = Dropout(dropouts[2])(hidden_layer)
# predictions = Dense(43, activation='softmax')(drop)
#
# model = Model(input=inputs, output=predictions)

## -- sequential

# inputs = Input(shape=(32, 32, 1))
#
# first_layer = Convolution2D(features[0], 3, 3, activation='relu')(inputs)
# #first_layer = Convolution2D(features[0], 3, 3, activation='relu')(first_layer)
#
# first_p_layer = MaxPooling2D(pool_size=(2, 2))(first_layer)
# drop_1 = Dropout(dropouts[0])(first_p_layer)
#
# second_p_layer = MaxPooling2D(pool_size=(2, 2))(drop_1)
#
# #first_input_layer = Flatten()(second_p_layer)
#
# second_layer = Convolution2D(features[1], 3, 3, activation='relu')(drop_1)
# #second_layer = Convolution2D(features[1], 3, 3, activation='relu')(second_layer)
#
# third_p_layer = MaxPooling2D(pool_size=(2, 2))(second_layer)
# drop_2 = Dropout(dropouts[1])(third_p_layer)
#
# second_input_layer = Flatten()(drop_2)
#
# #input_layer = merge([first_input_layer, second_input_layer], mode='concat', concat_axis=1)
# #hidden_layer = Dense(dense_hidden_units[0], activation='sigmoid')(input_layer)
# hidden_layer = Dense(dense_hidden_units[0], activation='sigmoid')(second_input_layer)
# drop = Dropout(dropouts[2])(hidden_layer)
# predictions = Dense(43, activation='softmax')(drop)
#
# model = Model(input=inputs, output=predictions)

# ----- end Sequential

def lr_schedule(epoch):
    return LR * (0.1 ** int(EPOCHS / 10))

if not eval_only:

    sgd = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                    metrics=['accuracy'],
                    loss='categorical_crossentropy')



    history_callback = model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_valid, y_valid)
            #   ,
            #   callbacks=[LearningRateScheduler(lr_schedule),
            #              ModelCheckpoint('{}.h5'.format(model_path), save_best_only=True)]
                  )

    loss_history = history_callback.history["loss"]
    log.log("loss:", False)
    log.log(loss_history, False)
    val_acc = history_callback.history["val_acc"]
    log.log("accuracy:", False)
    log.log(val_acc, False)

    time.sleep(0.1)

    log.log("Saving...", False)
    model_json = model.to_json()
    with open('{}.json'.format(model_path), "w") as json_file:
        json_file.write(model_json)
    model.save_weights('{}.h5'.format(model_path))
    log.log("Saved model to disk", False)


json_file = open('{}.json'.format(model_path), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('{}.h5'.format(model_path))
log.log("Loaded model from disk\n", False)
log.log(("Testing: "), False)
sgd = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)

loaded_model.compile(optimizer=sgd,
                metrics=['accuracy'],
                loss='categorical_crossentropy')

score = loaded_model.evaluate(X_test, y_test, verbose=1)
log.log('\nTest accuracy : ' + str(score[1]), False)

# predict and evaluate
# y_pred = model.predict_classes(X_test)
# acc = np.sum(y_pred == y_test) / np.size(y_pred)
# log.log("Test accuracy = {}".format(acc), False)

import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
import pickle

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.utils import shuffle
import random

import my_mod_load as load
import my_mod_manipulate_image as manipulate

DATASET_DIR = '/datasets/GTSRB/trainingSet_online/'
FINALTEST_DIR = '/datasets/GTSRB/testSet_online/Images/'
ANNOTATION_FILE = '/notebooks/signnames.csv'
FINAL_ANNOTATION_FILE = '/notebooks/GT-online_test.csv'
MANIPULATED_DIR = '/datasets/GTSRB/manipulated/'

IMAGE_SIZE = 32

parser = argparse.ArgumentParser(description="Traffic signs classifier")
parser.add_argument("-a", "--augmentation", help="Using augment data or not", action='store_true')
parser.add_argument("-b", "--blur", help='apply the blur function (augment data)', action='store_true')
parser.add_argument("--dataset", help='(online, pickle)', default='online')
parser.add_argument('--debug', help='Print debug messages', action='store_true')
parser.add_argument('--quiet', help='Print only the evaluation', action='store_true')
args = parser.parse_args()
#problem_name = args.problem
dataset_gtsrb = args.dataset

debug = True if args.debug else False
quiet = True if args.quiet else False
augmentation = True if args.augmentation else False
blur = True if args.blur else False


# -----------------------------
# manipulated dataset dir
# -----------------------------
date = time.strftime("%Y-%m-%d_%H%M")
folder = "{}_{}".format(date, dataset_gtsrb)
if augmentation :
    folder += "_augm"
if blur :
    folder += "_blur"

newpath = os.path.join(MANIPULATED_DIR, folder)
if not os.path.exists(newpath):
    os.makedirs(newpath)


#--------------------------------------------------
# Step 0: Load the Dataset
#--------------------------------------------------

if dataset_gtsrb == "online":
    #--------------------------------------------------
    # Load the Dataset from folders
    #--------------------------------------------------

    train = {}
    train['features'] = []
    train['labels'] = []
    valid = {}
    valid['features'] = []
    valid['labels'] = []
    load.load_train_valid_2(train, valid, DATASET_DIR, IMAGE_SIZE)
    # log.log("Dataset dimension on {} = {}".format(DATASET_DIR, len(dataset['features'])), False)
    # dataset_dim = len(dataset['features'])
    # conversione in np array
    train['features'] = np.array(train['features'])
    train['labels'] = np.array(train['labels'])
    valid['features'] = np.array(valid['features'])
    valid['labels'] = np.array(valid['labels'])

    test = {}
    test['features'] = []
    test['labels'] = []

    load.load_dataset_labeled_by_csv(test, FINALTEST_DIR, FINAL_ANNOTATION_FILE, ';', 'Filename', 'ClassId', IMAGE_SIZE)

    # conversione in np array
    test['features'] = np.array(test['features'])
    test['labels'] = np.array(test['labels'])

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']



elif dataset_gtsrb == "pickle":
    #-------------------------------------------------
    # Load the Dataset from pickle (.p files)
    #--------------------------------------------------

    import warnings
    warnings.filterwarnings('ignore')

    training_file = '/datasets/traffic-signs-data/train.p'
    validation_file= '/datasets/traffic-signs-data/valid.p'
    testing_file = '/datasets/traffic-signs-data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

n_train = len(X_train) # Number of training examples
n_test = len(X_test) # Number of testing examples.
n_valid = len(X_valid) # Number of testing examples.
n_classes = len(np.unique(y_train)) # How many unique classes/labels there are in the dataset.

# log.log("Number of training examples = {}".format(n_train), False)
# log.log("Number of validation examples = {}".format(n_valid), False)
# log.log("Number of testing examples = {}".format(n_test) , False)
# log.log("Number of classes = {}\n".format(n_classes), False)



#------------------------------------------------------------
# Step 2: Design and Test a Model Architecture
# Pre-process the Data Set (augmentation, blur, normalization, grayscale, etc.)
#------------------------------------------------------------

###### Step 2.1: augmentation (duplicate the X_train size)
if augmentation: # using keras ImageDataGenerator, work in progress
    print("Data augmentation\n")
    datagen = ImageDataGenerator(
        # perturbed in position ([-2, 2] pixels)
        width_shift_range=0.1,
        height_shift_range=0.1,
        # perturbed in scale ([.9, 1.1] ratio)
        #rescale=1./255,
        # perturbed in rotation ([-15, 15] degrees)
        rotation_range=15,
        shear_range=0.3,
        zoom_range=0.15,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest')

    # TODO aggiungere fattore moltiplicativo (esguendo più cicli?)
    # i cicli 2 e 3 creano immagini diverse dal primo?
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train)):
        X_train_aug = X_batch.astype('uint8')
        y_train_aug = y_batch
        break
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train)):
        X_train_aug2 = X_batch.astype('uint8')
        y_train_aug2 = y_batch
        break
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train)):
        X_train_aug3 = X_batch.astype('uint8')
        y_train_aug3 = y_batch
        break
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train)):
        X_train_aug4 = X_batch.astype('uint8')
        y_train_aug4 = y_batch
        break



###### Step 2.2: blurring (duplicate the X_train size)
if blur:
    X_train_br = list()
    y_train_br = list()

    for ii in range(len(X_train)):
        img = X_train[ii]
        label = y_train[ii]
        imgout = manipulate.sharpen_img(img)
        X_train_br.append(imgout)
        y_train_br.append(label)



###### Step 2.4: normalization and grayscale
X_train_norm = list()
y_train_norm = list()
X_test_norm = list()
X_valid_norm = list()

for ii in range(len(X_train)):
    img = X_train[ii]
    label = y_train[ii]
    imgout = manipulate.normalize_img(img)
    X_train_norm.append(imgout)
    y_train_norm.append(label)

train_path = os.path.join(newpath, "train.p")

X_train_norm = np.array(X_train_norm)
y_train_norm = np.array(y_train_norm)
# Step 2.5: saving manipulated dataset
d = {"features":X_train_norm.astype('float32'),"labels":y_train_norm}
with open(train_path, 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
del d

if augmentation:
    X_train_norm = list()
    y_train_norm = list()
    X_train2_norm = list()
    y_train2_norm = list()
    X_train3_norm = list()
    y_train3_norm = list()
    X_train4_norm = list()
    y_train4_norm = list()
    for ii in range(len(X_train_aug)):
        img = X_train_aug[ii]
        label = y_train_aug[ii]
        imgout = manipulate.normalize_img(img)
        X_train_norm.append(imgout)
        y_train_norm.append(label)

        img = X_train_aug2[ii]
        label = y_train_aug2[ii]
        imgout = manipulate.normalize_img(img)
        X_train2_norm.append(imgout)
        y_train2_norm.append(label)

        img = X_train_aug3[ii]
        label = y_train_aug3[ii]
        imgout = manipulate.normalize_img(img)
        X_train3_norm.append(imgout)
        y_train3_norm.append(label)

        img = X_train_aug4[ii]
        label = y_train_aug4[ii]
        imgout = manipulate.normalize_img(img)
        X_train4_norm.append(imgout)
        y_train4_norm.append(label)

    train_path = os.path.join(newpath, "train_aug1.p")
    X_train_norm = np.array(X_train_norm)
    y_train_norm = np.array(y_train_norm)
    # Step 2.5: saving manipulated dataset
    d = {"features":X_train_norm.astype('float32'), "labels":y_train_norm}
    with open(train_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del d

    train_path = os.path.join(newpath, "train_aug2.p")
    X_train2_norm = np.array(X_train2_norm)
    y_train2_norm = np.array(y_train2_norm)
    # Step 2.5: saving manipulated dataset
    d = {"features":X_train2_norm.astype('float32'), "labels":y_train2_norm}
    with open(train_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del d

    train_path = os.path.join(newpath, "train_aug3.p")

    X_train3_norm = np.array(X_train3_norm)
    y_train3_norm = np.array(y_train3_norm)
    # Step 2.5: saving manipulated dataset
    d = {"features":X_train3_norm.astype('float32'), "labels":y_train3_norm}
    with open(train_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del d

    train_path = os.path.join(newpath, "train_aug4.p")
    X_train4_norm = np.array(X_train4_norm)
    y_train4_norm = np.array(y_train4_norm)
    # Step 2.5: saving manipulated dataset
    d = {"features":X_train4_norm.astype('float32'), "labels":y_train4_norm}
    with open(train_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del d

if blur:
    X_train_norm = list()
    y_train_norm = list()
    for ii in range(len(X_train_br)):
        img = X_train_br[ii]
        label = y_train_br[ii]
        imgout = manipulate.normalize_img(img)
        X_train_norm.append(imgout)
        y_train_norm.append(label)

    train_path = os.path.join(newpath, "train_br1.p")
    X_train_norm = np.array(X_train_norm)
    y_train_norm = np.array(y_train_norm)
    # Step 2.5: saving manipulated dataset
    d = {"features":X_train_norm.astype('float32'), "labels":y_train_norm}
    with open(train_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del d

X_valid_norm = list()
y_valid_norm = list()
for ii in range(len(X_valid)):
    img = X_valid[ii]
    label = y_valid[ii]
    imgout = manipulate.normalize_img(img)
    X_valid_norm.append(imgout)
    y_valid_norm.append(label)


valid_path = os.path.join(newpath, "valid.p")
X_valid_norm = np.array(X_valid_norm)
y_valid_norm = np.array(y_valid_norm)
d = {"features":X_valid_norm.astype('float32'), "labels":y_valid_norm}
with open(valid_path, 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
del d

X_test_norm = list()
y_test_norm = list()
for ii in range(len(X_test)):
    img = X_test[ii]
    label = y_test[ii]
    imgout = manipulate.normalize_img(img)
    X_test_norm.append(imgout)
    y_test_norm.append(label)

test_path = os.path.join(newpath, "test.p")
X_test_norm = np.array(X_test_norm)
y_test_norm = np.array(y_test_norm)
d = {"features":X_test_norm.astype('float32'), "labels":y_test_norm}
with open(test_path, 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
del d

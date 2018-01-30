#!/usr/local/bin/python3

import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# import random


import my_mod_logs as log
import my_mod_load as load
import my_mod_manipulate_image as manipulate
import my_mod_nets as nets
#import my_mod_eval_predict as eval_pred
import time

DATASET_DIR = '/datasets/GTSRB/trainingSet_online/'
FINALTEST_DIR = '/datasets/GTSRB/testSet_online/Images/'
ANNOTATION_FILE = '/notebooks/signnames.csv'
FINAL_ANNOTATION_FILE = '/notebooks/GT-online_test.csv'

IMAGE_SIZE = 32

parser = argparse.ArgumentParser(description="Traffic signs classifier")
#parser.add_argument('problem', help='The problem name (inside ./in folder)')
parser.add_argument("-a", "--augmentation", help="Using augment data or not", action='store_true')
parser.add_argument("-b", "--batch_size", help='', default='128')
parser.add_argument("-d", "--dropout", help='', default='.3')
parser.add_argument("-e", "--epochs", help='The number of epochs', default='150')
parser.add_argument("-l", "--learning_rate", help='', default='1e-3')
parser.add_argument("-n", "--net", help='The net you wanna use (LeNet, LeNet-adv, VGGnet)', default='LeNet')
parser.add_argument("--dataset", help='(online, pickle)', default='online')
parser.add_argument('--debug', help='Print debug messages', action='store_true')
parser.add_argument('--quiet', help='Print only the evaluation', action='store_true')
args = parser.parse_args()
#problem_name = args.problem
net_name = args.net
EPOCHS = int(args.epochs)
LR = float(args.learning_rate)
BATCH_SIZE = int(args.batch_size)
dropout = float(args.dropout)
dataset_gtsrb = args.dataset

debug = True if args.debug else False
quiet = True if args.quiet else False
augmentation = True if args.augmentation else False

print("\nnet {}".format(net_name))
print("epochs: {}".format(EPOCHS))
print("learning rate: {}".format(LR))
print("bath size: {}".format(BATCH_SIZE))
print("dropout: {}".format(dropout))
# print(debug)
# print(quiet)
# print(augmentation)


# -----------------------------
# load the log file
# -----------------------------
log_file_name = "{}_{}_{}_{}_{}_{}_{}".format(time.strftime("%Y-%m-%d_%H%M"),
    net_name, EPOCHS, LR, BATCH_SIZE, dropout, dataset_gtsrb)

if augmentation :
    log_file_name += "_augm"

log.setup_file_logger('/logs/{}.log'.format(log_file_name))





#--------------------------------------------------
# Step 0: Load the Dataset
#--------------------------------------------------


if dataset_gtsrb == "online" :
    #--------------------------------------------------
    # Load the Dataset from folders
    #--------------------------------------------------

    trainingset = {}
    trainingset['features'] = []
    trainingset['labels'] = []
    validset= {}
    validset['features'] = []
    validset['labels'] = []
    load.load_trainset_validset_2(trainingset, validset, DATASET_DIR, IMAGE_SIZE)
    # log.log("Dataset dimension on {} = {}".format(DATASET_DIR, len(dataset['features'])), False)
    # dataset_dim = len(dataset['features'])
    testset = {}
    testset['features'] = []
    testset['labels'] = []

    load.load_dataset_labeled_by_csv(testset, FINALTEST_DIR, FINAL_ANNOTATION_FILE, ';', 'Filename', 'ClassId', IMAGE_SIZE)

    X_train, y_train = trainingset['features'], trainingset['labels']
    X_valid, y_valid = validset['features'], validset['labels']
    X_test, y_test = testset['features'], testset['labels']

elif dataset_gtsrb == "pickle" :
    #-------------------------------------------------
    # Load the Dataset from pickle (.p files)
    #--------------------------------------------------

    import pickle
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

log.log("Number of training examples = {}".format(n_train), False)
log.log("Number of validation examples = {}".format(n_valid), False)
log.log("Number of testing examples = {}".format(n_test) , False)
log.log("Number of classes = {}\n".format(n_classes), False)




#------------------------------------------------------------
# Step 2: Design and Test a Model Architecture
# Pre-process the Data Set (normalization, grayscale, etc.)
#------------------------------------------------------------

# Transform all images and augment training data
if augmentation:
    print("Data augmentation\n")

    # -----
    # augmentation using keras, work in progress
    # -----

    # datagen = ImageDataGenerator(
    #     rotation_range=17,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     shear_range=0.3,
    #     zoom_range=0.15,
    #     horizontal_flip=False,
    #     fill_mode='nearest')
    #
    # for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False):
    #     # print(X_batch.shape)
    #     X_train_aug = X_batch.astype('uint8')
    #     y_train_aug = y_batch
    #     #img_out_rgb = cv2.cvtColor(X_batch[0].astype('float32'), cv2.COLOR_BGR2RGB);
    #     #cv2.imwrite("out.jpg",img_out_rgb)
    #     # create a grid of 3x3 images
    #     break



X_train_transf = list()
y_train_transf = list()
X_test_transf = list()
X_valid_transf = list()

for ii in range(len(X_train)):
    img = X_train[ii]
    label = y_train[ii]

    imgout = manipulate.normalize_img(img)
    X_train_transf.append(imgout)
    y_train_transf.append(label)

    if augmentation:
        for j in range(4):
            imgout = manipulate.augment_img(img)
            X_train_transf.append(imgout)
            y_train_transf.append(label)

for ii in range(len(X_valid)):
    img = X_valid[ii]
    img = manipulate.normalize_img(img)
    X_valid_transf.append(img)

for ii in range(len(X_test)):
    img = X_test[ii]
    img = manipulate.normalize_img(img)
    X_test_transf.append(img)





if augmentation:
    n_train = len(X_train_transf) # Number of training examples
    log.log("Number of training examples (augmentated) = {}".format(n_train), False)

#Definizione dei placeholder
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
keep_prob_conv = tf.placeholder(tf.float32)

#Restituisce un tensore (a valori binari) contenente valori tutti posti a 0 tranne uno.
one_hot_y = tf.one_hot(y, 43)



#Variabili necessarie per la fase di training e di testing
if net_name == 'LeNet':
    logits = nets.LeNet(x, keep_prob)
    log.log("used net = LeNet", False)
elif net_name == 'LeNet-adv':
    logits = nets.LeNet-adv(x, keep_prob)
    log.log("used net = LeNet-adv", False)
elif net_name == 'VGGnet':
    logits = nets.VGGnet(x, keep_prob, keep_prob_conv)
    log.log("used net = VGGnet", False)
else:
    sys.exit()



#softmax_cross_entropy_with_logits(_sentinel, labels, logits, dim, name)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = LR)
training_operation = optimizer.minimize(loss_operation)
predict_operation = tf.argmax(logits, 1)
predict_proba_operation = tf.nn.softmax(logits=logits)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def predict(X_data):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    predicted_proba = list()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x = X_data[offset:offset+BATCH_SIZE]
        predicted_proba.extend( sess.run(predict_proba_operation, feed_dict={x: batch_x, keep_prob: 1.0, keep_prob_conv:1}))


    return predicted_proba


#------------------------------------------------------------------------------
# Training
#------------------------------------------------------------------------------

from sklearn.utils import shuffle
from time import time

X_train = X_train_transf
X_valid = X_valid_transf
X_test = X_test_transf
y_train = y_train_transf

#EPOCHS = 150
#BATCH_SIZE = 128
#dropout = .3

errors = list()

saver = tf.train.Saver()
start = time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    log.log("Training... dropout = {} , batch_size = {} , learning rate = {}".format(dropout, BATCH_SIZE, LR), True)
    print()
    for i in range(EPOCHS):
        try:
            X_train, y_train = shuffle(X_train, y_train)
#             print("Before Train %d sec"%(time() - start))

            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1 - dropout, keep_prob_conv: 0.7})
                # sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5 , keep_prob_conv: 0.7})

#           print("After Train %d sec"%(time() - start))

            validation_accuracy = evaluate(X_valid, y_valid)
            training_accuracy = evaluate(X_train, y_train)

            errors.append((training_accuracy, validation_accuracy))

#           calculatiing minutes format
            minutes = int((time() - start)/60)
            seconds = ((time() - start) - (minutes*60))

            log.log("EPOCH %d - %d sec ....%d.%d min"%(i+1, time() - start,  minutes, seconds), True)
            log.log("Training error = {:.3f} Validation error = {:.3f}".format(1- training_accuracy , 1- validation_accuracy), True)

            print()

#             print("After error computation %d sec"%(time() - start))
            if i > 5 and i % 3 == 0:
                saver.save(sess, './models/lenet')
                print("Model saved %d sec"%(time() - start))
        except KeyboardInterrupt:
            print('Accuracy Model On Test Images: {}'.format(evaluate(X_test,y_test)))
            break

    saver.save(sess, './models/lenet')






#Printing accuracy of the model on Training, validation and Testing set.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./models'))
    log.log("ACCURACY:", False)
    log.log('Accuracy Model On Training Images: {:.3f}'.format(evaluate(X_train, y_train)), False)
    log.log('Accuracy Model On Validation Images: {:.3f}'.format(evaluate(X_valid, y_valid)), False)
    log.log('Accuracy Model On Test Images: {:.3f}'.format(evaluate(X_test, y_test)), False)

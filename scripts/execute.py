#!/usr/local/bin/python3

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import my_mod_logs as log
import my_mod_load as load
import my_mod_manipulate_image as manipulate
import my_mod_nets as nets
#import my_mod_eval_predict as eval_pred
import time

DATASET_DIR = '/datasets/GTSRB/Images/'
FINALTEST_DIR = '/datasets/GTSRB/Final_Test/Images/'
ANNOTATION_FILE = '/notebooks/signnames.csv'
FINAL_ANNOTATION_FILE = '/notebooks/GT-final_test.csv'

IMAGE_SIZE = 32
LR = 1e-3

parser = argparse.ArgumentParser(description="Traffic signs classifier")
#parser.add_argument('problem', help='The problem name (inside ./in folder)')
parser.add_argument("--net", help='The net you wanna use (LeNet, )', default='LeNet')
parser.add_argument("--epoch", help='The number of epoch', default='150')
parser.add_argument("--learning_rate", help='', default='1e-3')
parser.add_argument('--debug', help='Print debug messages', action='store_true')
parser.add_argument('--quiet', help='Print only the evaluation', action='store_true')

args = parser.parse_args()
#problem_name = args.problem
net_name = args.net
epoch_num = args.epoch
debug = '--debug' if args.debug else ''

print(net_name)
print(debug)



# -----------------------------
# log module test
# -----------------------------
log.setup_file_logger('/logs/{}_training.log'.format(time.strftime("%Y-%m-%d_%H%M")))

# -----------------------------




#--------------------------------------------------
# Step 0: Load the Dataset
#--------------------------------------------------

from sklearn.model_selection import train_test_split

dataset = {}
dataset['features'] = []
dataset['labels'] = []
load.load_dataset_labeled_by_dirs(dataset, DATASET_DIR, IMAGE_SIZE)
log.log("Dataset dimension on {} = {}".format(DATASET_DIR, len(dataset['features'])), False)
dataset_dim = len(dataset['features'])
load.load_dataset_labeled_by_csv(dataset, FINALTEST_DIR, FINAL_ANNOTATION_FILE, ';', 'Filename', 'ClassId', IMAGE_SIZE)
log.log("Dataset dimension on {} = {}".format(FINALTEST_DIR, len(dataset['features'])- dataset_dim), False)
log.log("Final dimension = {}".format(len(dataset['features'])), False)
#np.save('dataset.npy', dataset)

X, y = dataset['features'], dataset['labels']
#prende il 70% per il train e il 30% per il vaild
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.2)

n_train = len(X_train) # Number of training examples
n_test = len(X_test) # Number of testing examples.
n_valid = len(X_valid) # Number of testing examples.
n_classes = len(np.unique(y_train)) # How many unique classes/labels there are in the dataset.

log.log("\tNumber of training examples = {}".format(n_train), False)
log.log("\tNumber of validation examples = {}".format(n_valid), False)
log.log("\tNumber of testing examples = {}".format(n_test) , False)
log.log("Number of classes = {}".format(n_classes), False)




#------------------------------------------------------------
# Step 2: Design and Test a Model Architecture
# Pre-process the Data Set (normalization, grayscale, etc.)
#------------------------------------------------------------

# Transform all images and augment training data
X_train_transf = list()
y_train_transf = list()
X_test_transf = list()
X_valid_transf = list()
for ii in range(len(X_train)):
    img = X_train[ii]
    label = y_train[ii]

    imgout = manipulate.transform_img(img)
    imgout.shape = imgout.shape + (1,)
    X_train_transf.append(imgout)
    y_train_transf.append(label)

for ii in range(len(X_valid)):
    img = X_valid[ii]
    img = manipulate.transform_img(img)
    img.shape = img.shape + (1,)
    X_valid_transf.append(img)

for ii in range(len(X_test)):
    img = X_test[ii]
    img = manipulate.transform_img(img)
    img.shape = img.shape + (1,)
    X_test_transf.append(img)



#Definizione dei placeholder
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

#Restituisce un tensore (a valori binari) contenente valori tutti posti a 0 tranne uno.
one_hot_y = tf.one_hot(y, 43)



#Variabili necessarie per la fase di training e di testing

logits = nets.LeNet(x, keep_prob)
log.log("used net = LeNet", False)
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
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def predict(X_data):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    predicted_proba = list()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x = X_data[offset:offset+BATCH_SIZE]
        predicted_proba.extend( sess.run(predict_proba_operation, feed_dict={x: batch_x, keep_prob: 1.0}))


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
EPOCHS = int(epoch_num)
BATCH_SIZE = 128
dropout = .3

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
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1 - dropout})

#             print("After Train %d sec"%(time() - start))

            validation_accuracy = evaluate(X_valid, y_valid)
            training_accuracy = evaluate(X_train, y_train)

            errors.append((training_accuracy,validation_accuracy))
            log.log("EPOCH %d - %d sec ..."%(i+1, time() - start), True)
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

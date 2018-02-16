import sys

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
import settings






def my_test(model, learning_rate, batch_size):

    LR = learning_rate
    BATCH_SIZE = batch_size

    list = []

    images, labels_wild = load.load_new_data()

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

    predicted_proba = loaded_model.predict(images)

    well_aimed = 0
    num_images = 0
    for true_label,row in zip(labels_wild, predicted_proba):
        top3k = np.argsort(row)[::-1][:3]
        top3p = np.sort(row)[::-1][:3]
        if(true_label == top3k[0]):
            well_aimed += 1
        # print('Top 3 Labels for image \'{}\':'.format(load.get_name_from_label(true_label)))
        list.append((images[num_images], true_label, top3k, top3p))
        # for k,p in zip(top3k,top3p):
            #   print(' - \'{}\' with prob = {:.4f} '.format(load.get_name_from_label(k), p))
        # print()
        num_images += 1

    print("well-aimed: {}/{}".format(well_aimed, num_images))
    print("well-aimed: {:.4f}".format(well_aimed/num_images))
    return list

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import my_mod_manipulate_image as mod
import numpy as np
import my_mod_load as mml
import my_mod_test_new as test_new
import settings
import sys
import argparse
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

def original_vs_normalized(img): 
    fig = plt.figure(figsize=(12, 4))

    #immagine originale
    ax1 = fig.add_subplot(131)
    plt.imshow(img)
    ax1.set_title('Original',fontsize=16)
    plt.axis('off')

    #manipolazione dell'immagine
    img_norm = mod.normalize_img(img)

    #immagine normalizzata
    ax2 = fig.add_subplot(132)
    plt.imshow(img_norm[:,:,0], cmap='gray')
    ax2.set_title('Normalized',fontsize=16)
    plt.axis('off')

    plt.show()
    
    
def orginal_vs_blurred(img):
    fig = plt.figure(figsize=(12, 4))

    #immagine originale
    ax1 = fig.add_subplot(131)
    plt.imshow(img)
    ax1.set_title('Original',fontsize=16)
    plt.axis('off')

    #manipolazione dell'immagine
    img_blur = mod.motion_blur(img)
    img_blur_norm = mod.normalize_img(img_blur) 

    #immagine normalizzata
    ax2 = fig.add_subplot(132)
    plt.imshow(img_blur_norm[:,:,0], cmap='gray')
    ax2.set_title('Blurred-Normalized',fontsize=16)
    plt.axis('off')

    plt.show()
    
    
def final_image(model):
    
    l , n_examples, list_fname = test_new.my_test(model, learning_rate=0.01, batch_size=128)
    fig, ax = plt.subplots(n_examples, 2,figsize=(12,35))

    for i in range(n_examples):
        labels = l[i][1]
        img = mpimg.imread(list_fname[i])
        names = mml.get_name_from_label(l[i][2])
        bar_locations = np.arange(3)[::-1]
        ax[i,0].imshow(img)
        ax[i,0].axis('off')
        ax[i,1].barh(bar_locations, l[i][3])
        ax[i,1].set_yticks(0.1+bar_locations)
        ax[i,1].set_yticklabels(names)
        ax[i,1].yaxis.tick_right()
        ax[i,1].set_xlim([0,1])
    ax[0,1].set_title('Model Prediction')
    plt.tight_layout()
    plt.show()
    
def bad_aimed(model):
    count = 0
    index = []
    l , n_examples, list_fname = test_new.my_test(model, learning_rate=0.01, batch_size=128)

    for i in range(n_examples):
        if l[i][1] != l[i][2][0]:
            index.append(i)
            count += 1

    fig, ax = plt.subplots(count, 2,figsize=(15,8))

    for ii in range(count):
        i = index[ii]
        labels = l[i][1]
        img = mpimg.imread(list_fname[i])
        names = mml.get_name_from_label(l[i][2])
        bar_locations = np.arange(3)[::-1]
        ax[ii,0].imshow(img)
        ax[ii,0].axis('off')
        ax[ii,1].barh(bar_locations, l[i][3])
        ax[ii,1].set_yticks(0.1+bar_locations)
        ax[ii,1].set_yticklabels(names)
        ax[ii,1].yaxis.tick_right()
        ax[ii,1].set_xlim([0,1])
    ax[0,1].set_title('Model Prediction')
    plt.tight_layout()
    plt.show()
    
def bad_test_aimed(model, X_test_nm):
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
    #print("Loaded model from disk\n")
    #print("Testing: ")
    sgd = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)

    loaded_model.compile(optimizer=sgd,
                    metrics=['accuracy'],
                    loss='categorical_crossentropy')


    predicted_proba = loaded_model.predict(X_test)

    labels_wild = y_test

    well_aimed = 0
    num_images = 0
    X_bad_aimed = list()
    for true_label,row in zip(labels_wild, predicted_proba):
        num_images += 1
        topk = np.argsort(row)[::-1][:1]
        topp = np.sort(row)[::-1][:1]
        if(true_label == topk[0]):
            well_aimed += 1
        else:
            X_bad_aimed.append(X_test_nm[num_images])
            
    #plotting image bad aimed        
    fig, axes = plt.subplots(8, 11, figsize=(15, 8))
    ii = 0
    for ax in axes.flatten() :
        ax.imshow(X_bad_aimed[ii])
        ax.axis('off')
        ii += 1
    plt.show()
   
    #print("well-aimed: {}/{}".format(well_aimed, num_images))
    #print("well-aimed: {:.4f}".format(well_aimed/num_images))
    return X_bad_aimed
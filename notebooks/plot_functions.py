import os
import re
import settings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import my_mod_manipulate_image as mod
import numpy as np
import my_mod_load as mml
import my_mod_test_new as test_new

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



def learning_curve(last=False, file_name=""):

    if last == False:
        log_file_path = os.path.join(settings.LOG_DIR, file_name)
    else:
        for root, dirs, files in os.walk(settings.LOG_DIR):
            for log_file in sorted(files, reverse=True):
                log_file_path = os.path.join(settings.LOG_DIR, log_file)
                break
            break

    with open(log_file_path) as inputFile:
        lines = inputFile.readlines()

    strings = ("INFO accuracy", "xINFO loss", "string3")

    accuracy = []
    val_accuracy = []
    loss = []
    next_one_acc = False
    next_one_loss = False
    next_one_val= False

    for line in lines:
        if next_one_acc:
            matchObj = re.match( r'(.*) INFO \[(.*)\]', line, re.M|re.I)
            # print(matchObj.group(2))
            accuracy = [x.strip() for x in matchObj.group(2).split(',')]
            # print(accuracy)
            next_one_acc = False
        if next_one_val:
            matchObj = re.match( r'(.*) INFO \[(.*)\]', line, re.M|re.I)
            # print(matchObj.group(2))
            val_accuracy = [x.strip() for x in matchObj.group(2).split(',')]
            # print(val_accuracy)
            next_one_val = False
        if next_one_loss:
            matchObj = re.match( r'(.*) INFO \[(.*)\]', line, re.M|re.I)
            # print(matchObj.group(2))
            loss = [x.strip() for x in matchObj.group(2).split(',')]
            # print(loss)
            next_one_loss = False
        # if any(s in line for s in strings):
        #     print(s)
        if "INFO accuracy" in line:
            next_one_acc = True
        if "INFO val accuracy" in line:
            next_one_val = True
        if "INFO loss" in line:
            next_one_loss = True

    plt.figure(figsize=(8,6))
    plt.title('Learning Curve')
    #plt.plot([el for el in loss])
    plt.plot([1 - float(el) for el in accuracy])
    if len(val_accuracy) != 0:
        plt.plot([1 - float(el) for el in val_accuracy])
        plt.legend(['Training error', 'Validation error'])
    else:
        plt.legend((['Training error']))
    plt.tight_layout()
    plt.savefig('images/learning_curve.png')
    plt.ylabel('Error')
    plt.xlabel('Epoch');

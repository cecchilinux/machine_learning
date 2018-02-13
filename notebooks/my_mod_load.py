import cv2 # resize the images
import numpy as np
import pandas as pd
import os # to work with directories
import csv
import random
import pickle

import settings

signnames_path = './signnames.csv'
signnames = pd.read_csv(signnames_path)
signnames.set_index('ClassId',inplace=True)


def get_classes():
    list = []
    with open(signnames_path, 'r') as data_file:
        data_file.readline() # Skip first line
        reader = csv.reader(data_file, delimiter=',')

        # per ogni file della classe
        for classId, SignName in reader:
            #print(classId, SignName)
            list.append(classId + ": " + SignName)
    return(list)


def get_name_from_label(label):
    # Helper, transofrm a numeric label into the corresponding strring
    return signnames.loc[label].SignName



def load_manipulated_train(dirname):
    train_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train.p")
    train_aug1_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_aug1.p")
    train_aug2_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_aug2.p")
    train_aug3_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_aug3.p")
    train_aug4_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_aug4.p")
    train_br1_path = os.path.join(settings.MANIPULATED_DIR, dirname, "train_br1.p")

    with open(train_path, mode='rb') as f:
        train = pickle.load(f)
    X_train, y_train = train['features'], train['labels']

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

    with open(train_br1_path, mode='rb') as f:
        train_br1 = pickle.load(f)

    X_train_br, y_train_br = train_br1['features'], train_br1['labels']

    X_train = np.concatenate((X_train, X_train_br), axis=0)
    y_train = np.concatenate((y_train, y_train_br), axis=0)

    return X_train, y_train


def load_normalized_valid(dirname):
    valid_path = os.path.join(settings.MANIPULATED_DIR, dirname, "valid.p")
    with open(valid_path, mode='rb') as f:
        valid = pickle.load(f)
    X_valid, y_valid = valid['features'], valid['labels']
    return X_valid, y_valid


def load_normalized_test(dirname):
    test_path = os.path.join(settings.MANIPULATED_DIR, dirname, "test.p")
    with open(test_path, mode='rb') as f:
        test = pickle.load(f)
    X_test, y_test = test['features'], test['labels']
    return X_test, y_test


def load_train_valid_test():
    train, valid = load_train_valid(settings.DATASET_DIR, settings.IMAGE_SIZE)
    test = load_dataset_labeled_by_csv(settings.TEST_DIR, settings.TEST_ANNOTATION_FILE, ';', 'Filename', 'ClassId', settings.IMAGE_SIZE)
    # return {'train':train, 'valid':valid,'test':test}
    return train, valid, test

def load_train_valid(path, image_size):
    train = {}
    train['features'] = []
    train['labels'] = []
    valid = {}
    valid['features'] = []
    valid['labels'] = []

    for root, dirs, files in os.walk(path):

        #per ogni classe
        for dirname in sorted(dirs):
            subdir_path = os.path.join(path, dirname)

            in_filename = "{}/GT-{}.csv".format(subdir_path, dirname)
            #reader = csv.reader(open(in_filename), delimiter=';')
            d = {} # dizionario in cui la chiave è la track e il valore è la lista dei nomi delle immagini

            #leggo il csv
            with open(in_filename, 'r') as data_file:
                data_file.readline() # Skip first line
                reader = csv.reader(data_file, delimiter=';')
                trackName = ""

                # per ogni file della classe
                for name, _, _, _, _, _, _, classId in reader:
                    track = name[:5] # nome della track corrente
                    if(trackName != track): # se è una track nuova
                        d[track] = [] # inizializzo la lista per la nuova track
                        trackName = track
                    d[track].append(name) # appendo l'immagine alla sua track
                    #print(classId + "  ," + name[:5] + " ," + name)

                n_track = len(d.keys()) # numero di track per la classe
                #estraggo un numero random tra 0 e il numero di track
                track_rnd = random.randint(0,n_track-1)

                #controllo per risalire all'etichetta parziale della track
                if track_rnd < 10:
                    str_track_rnd = '0000'+str(track_rnd)
                else:
                    str_track_rnd = '000'+str(track_rnd)

                # scorro il dizionario track by track
                for key in d:
                    # per ogni immagine
                    for imgName in d[key]:
                        label = int(os.path.basename(subdir_path))
                        imgPath = os.path.join(subdir_path, imgName)
                        img = cv2.resize(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB), (image_size, image_size))
                        if(key != str_track_rnd):
                            train['features'].append(np.asarray(img))
                            train['labels'].append(label)
                        else:
                            valid['features'].append(np.asarray(img))
                            valid['labels'].append(label)

    # conversione in np array
    train['features'] = np.array(train['features'])
    train['labels'] = np.array(train['labels'])
    valid['features'] = np.array(valid['features'])
    valid['labels'] = np.array(valid['labels'])

    return train, valid



# def load_train_valid_2(train, valid, path, image_size):
#
#
#     for root, dirs, files in os.walk(path):
#
#         #per ogni classe
#         for dirname in sorted(dirs):
#             subdir_path = os.path.join(path, dirname)
#
#             in_filename = "{}/GT-{}.csv".format(subdir_path, dirname)
#             #reader = csv.reader(open(in_filename), delimiter=';')
#             d = {} # dizionario in cui la chiave è la track e il valore è la lista dei nomi delle immagini
#
#             #leggo il csv
#             with open(in_filename, 'r') as data_file:
#                 data_file.readline() # Skip first line
#                 reader = csv.reader(data_file, delimiter=';')
#                 trackName = ""
#
#                 # per ogni file della classe
#                 for name, _, _, _, _, _, _, classId in reader:
#                     track = name[:5] # nome della track corrente
#                     if(trackName != track): # se è una track nuova
#                         d[track] = [] # inizializzo la lista per la nuova track
#                         trackName = track
#                     d[track].append(name) # appendo l'immagine alla sua track
#                     #print(classId + "  ," + name[:5] + " ," + name)
#
#                 n_track = len(d.keys()) # numero di track per la classe
#                 #estraggo un numero random tra 0 e il numero di track
#                 track_rnd = random.randint(0,n_track-1)
#
#                 #controllo per risalire all'etichetta parziale della track
#                 if track_rnd < 10:
#                     str_track_rnd = '0000'+str(track_rnd)
#                 else:
#                     str_track_rnd = '000'+str(track_rnd)
#
#                 # scorro il dizionario track by track
#                 for key in d:
#                     # per ogni immagine
#                     for imgName in d[key]:
#                         label = int(os.path.basename(subdir_path))
#                         imgPath = os.path.join(subdir_path, imgName)
#                         img = cv2.resize(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB), (image_size, image_size))
#                         if(key != str_track_rnd):
#                             train['features'].append(np.asarray(img))
#                             train['labels'].append(label)
#                         else:
#                             valid['features'].append(np.asarray(img))
#                             valid['labels'].append(label)




def load_dataset_labeled_by_dirs(dataset, path, image_size):
    """Load a dataset of images divided by folders

    this function look for images on the subfolders of the given path and label
    them with the name of the folder where the image is stored

    Parameters
    ----------
    dataset : the dictionary where to add the images
    path : the path where the images divided into folders are stored

    Returns
    -------

    """

    for subdir, dirs, files in os.walk(path): # all file on the dataset folder
        for file in files: # one image by one

            _, file_extension = os.path.splitext(file) # extension control
            if file_extension == '.ppm':
                label = os.path.basename(subdir) # obtain the image label (name of the folder)
                imgPath = os.path.join(path, label, file) # the path of the image

                # load image with cv2 library
                img = cv2.resize(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB), (image_size, image_size))
                label = int(label) # remove the zeros ahead the name of the folder

                dataset['features'].append(np.asarray(img))
                dataset['labels'].append(label)



def load_dataset_labeled_by_csv(path, file_csv, delimiter, index_col, label_col, image_size):
    """Load a dataset of images labeled with a csv file

    this function look for images on the subfolders of the given path and label
    them with the corrisponding label stored on a csv file

    Parameters
    ----------
    path : the path where the images divided into folders are stored
    file_csv : the path of the csv file
    delimiter : the delimeter used to separate coloumns of the csv file
    index_col : the name of the index coloumn
    label_col : the name of the label coloumn

    Returns
    -------

    """

    dataset = {}
    dataset['features'] = []
    dataset['labels'] = []


    final_test_csv = pd.read_csv(file_csv, delimiter=delimiter, encoding="utf-8-sig")
    final_test_csv.set_index(index_col, inplace=True)

    for subdir, dirs, files in os.walk(path): # all file on the dataset folder
        for file in files: # one image by one

            filename, file_extension = os.path.splitext(file) # extension control
            if file_extension == '.ppm':
                label = final_test_csv.loc[filename + file_extension].ClassId # obtain the image label
                imgPath = os.path.join(path, file) # the path of the image

                # load image with cv2 library
                img = cv2.resize(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB), (image_size, image_size))

                dataset['features'].append(np.asarray(img))
                dataset['labels'].append(label)

    return dataset


import skimage
from skimage import io
from skimage import transform
from skimage.filters import gaussian
import my_mod_manipulate_image as manipulate

def load_new_data():

    # Read the images
    i=1
    images_wild = list()
    labels_wild = list()
    for line in open('./Sign_image_resized/data.txt','r'):
        fname, label = line.strip().split(' ')
        label = int(label)
        fname = './Sign_image_resized/'+fname
        img = io.imread(fname)
        img = transform.resize(img,(32,32), order=3)
        img = gaussian(img,.6,multichannel=True)*255
        #img = transform_img(img.astype(np.uint8))
        img = manipulate.normalize_img(img.astype(np.uint8))

        img.shape = (1,) + img.shape
        images_wild.append(img)
        labels_wild.append(label)

    images = np.concatenate(images_wild, axis=0)
    return images, labels_wild


def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)


### Data exploration / visualization
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

import matplotlib.image as mpimg
import random

def split_by_class(y_data) :
    """
    Returns a dictionary whose keys are the class labels
    and key values are list of indices with that class label.
    """
    img_index = {}
    labels = set(y_data)
    for i,y in enumerate(y_data) :
        if y not in img_index.keys() :
            img_index[y] = [i]
        else :
            img_index[y].append(i)
    return img_index

def display_sample_images(X_data, y_data, i_start='random') :
    """
    Displays images from each class,
    i_start is the starting index in each class.
    By default, images are randomly chosen from each class
    """
    img_index = split_by_class(y_data)
    labels = list(set(y_data))[::-1]
    fig, axes = plt.subplots(3, 15, figsize=(15, 3))
    for ax in axes.flatten() :
        if labels :
            i_img=0
            if i_start == 'random':
                i_img = random.choice(img_index[labels.pop()])
            else :
                i_img = img_index[labels.pop()][i_start]
            ax.imshow(X_data[i_img])
        ax.axis('off')


def display_class_distribution(y_data):
    sign_classes, class_indices, class_counts = np.unique(y_data, return_index = True, return_counts = True)

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(43), class_counts, align='center')
    plt.xlabel('Classe')
    plt.ylabel('Numero di immagini')
    plt.xlim(-1, 43)
    plt.show()

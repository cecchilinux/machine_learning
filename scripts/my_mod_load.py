import cv2 # resize the images
import numpy as np
import pandas as pd
import os # to work with directories
import csv
import random

def row_count(filename):
    with open(filename) as in_file:
        return sum(1 for _ in in_file)


def load_trainset_validset(trainingset, validset, path, image_size):
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

    for root, dirs, files in os.walk(path):
        for dirname in sorted(dirs):
            subdir_path = os.path.join(path, dirname)

            in_filename = "{}/GT-{}.csv".format(subdir_path, dirname)
            reader = csv.reader(open(in_filename), delimiter=';')

            last_line_number = row_count(in_filename)
            for row in reader:
                if last_line_number == reader.line_num:

                    #aggiungo +1 per stampare il numero corretto di track
                    n_track = int(row[0][:5])+1
                    print("classe:{}\tnum track:{}".format(dirname, n_track))

            #estraggo un numero random tra 0 e il numero di track
            track_rnd = random.randint(0,n_track-1)

            #controllo per risalire all'etichetta parziale della track
            if track_rnd < 10:
                str_track_rnd = '0000'+str(track_rnd)
            else:
                str_track_rnd = '000'+str(track_rnd)
            print("Random track: {}".format(str_track_rnd))

            for f in os.walk(subdir_path):
                #creo un array contenente i nomi dei file .ppm
                file_list = sorted(f[2])

                #ciclo che controlla tutte i file .ppm e controlla se coincidono con la track estratta
                for item in file_list:
                    label = int(dirname)
                    imgPath = os.path.join(subdir_path, item)
                    img = cv2.resize(cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB), (image_size, image_size))
                    #Se le prime 5 cifre del file coincidono con la track allora --_>  validset
                    if item[:5] == str_track_rnd:
                        validset['features'].append(np.asarray(img))
                        validset['labels'].append(label)
                    #per tutto il resto c'è mastercard....
                    #...il resto viene campionato come training set
                    else:
                        trainingset['features'].append(np.asarray(img))
                        trainingset['labels'].append(label)
                        #assign
                # ATTENZIONE: exit() per evitare di far eseguire tutti i cicli..
                #  .. per la computazione completa RIMUOVERLO
                exit()


def load_trainset_validset_2(trainingset, validset, path, image_size):


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
                            trainingset['features'].append(np.asarray(img))
                            trainingset['labels'].append(label)
                        else:
                            validset['features'].append(np.asarray(img))
                            validset['labels'].append(label)

                


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



def load_dataset_labeled_by_csv(dataset, path, file_csv, delimiter, index_col, label_col, image_size):
    """Load a dataset of images labeled with a csv file

    this function look for images on the subfolders of the given path and label
    them with the corrisponding label stored on a csv file

    Parameters
    ----------
    dataset : the dictionary where to add the images
    path : the path where the images divided into folders are stored
    file_csv : the path of the csv file
    delimiter : the delimeter used to separate coloumns of the csv file
    index_col : the name of the index coloumn
    label_col : the name of the label coloumn

    Returns
    -------

    """

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

import cv2 # resize the images
import numpy as np
import pandas as pd
import os # to work with directories


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

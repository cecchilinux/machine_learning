import sys
import skimage
from skimage import io
from skimage import transform
from skimage.filters import gaussian

from keras.layers import Dense, Dropout, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import my_mod_load as load
import my_mod_manipulate_image as manipulate
import numpy as np
import settings


from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.engine import InputLayer
import keras

def to_fully_conv(model):

    new_model = Sequential()

    input_layer = InputLayer(input_shape=(None, None, 1), name="input_new")

    new_model.add(input_layer)

    for layer in model.layers:

        if "Flatten" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape
            print("flatten")

        elif "Dense" in str(layer):

            input_shape = layer.input_shape
            output_dim =  layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],f_dim[3],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (f_dim[1],f_dim[2]),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b],
                                          name=layer.name+"_x")
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b],
                                          name=layer.name+"_x")


        else:
            new_layer = layer

        print(layer.name + "   " + new_layer.name)
        new_model.add(new_layer)

    return new_model

model = '/notebooks/final_models/keras/2018-02-23_1203/model_108-108-100_ep40'
# model = './models/2018-02-19_2104/model_108-108-100_ep80'

LR = 0.01

my_list = []

# image, labels = load.load_image('', '')
fname = '/datasets/example3.png'
img = io.imread(fname)
img = transform.resize(img,(224, 224), order=3)
img = gaussian(img,.6,multichannel=True)*255
#img = transform_img(img.astype(np.uint8))
img = manipulate.normalize_img(img.astype(np.uint8))

img.shape = (1,) + img.shape
# images_wild.append(img)



json_file = open("{}.json".format(model), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("{}.h5".format(model))

print(loaded_model.summary())

# sys.exit()

loaded_model_2 = to_fully_conv(loaded_model)

print("Loaded model from disk\n")
print("Testing: ")

# sgd = SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
#
# loaded_model.compile(optimizer=sgd,
#                 metrics=['accuracy'],
#                 loss='categorical_crossentropy')

predicted_proba = loaded_model_2.predict(img)
print(predicted_proba.shape)
#print(predicted_proba)
print(predicted_proba.shape[2])
num = predicted_proba.shape[2]
m = 0
for i in range(0, num):
    for j in range(0, num):
        top1 = np.sort(predicted_proba[0][i][j])[42]
        if(top1 >= 0.8):
            print(top1)
            m+=1

print(m)
# for i in range(0, predicted_proba)
# for row in predicted_proba:
#     top3p = np.sort(row)[::-1][:3]

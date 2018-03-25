import numpy as np
import cv2
from skimage import exposure # a collection of algorithms for image processing.
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# normalization
# -----------------------

def normalize_img(img):

    img_y = cv2.cvtColor(img, (cv2.COLOR_RGB2YUV))[:,:,0] #converte l'immagine in YUV e tiene il canale Y

    # ----- global equalization
    img_y = cv2.equalizeHist(img_y)

    img_y = (img_y / 255.).astype(np.float32) # rappresenta i valori in un range di [0-1]


    # ----- local equalization

    # An algorithm for local contrast enhancement, that uses histograms computed
    # 0 over different tile regions of the image. Local details can therefore be
    # enhanced even in regions that are darker or lighter than most of the image.
    img_y = (exposure.equalize_adapthist(img_y) - 0.5)

    img_y = img_y.reshape(img_y.shape + (1,))

    return img_y

def normalize_img_color(img):
    img = cv2.cvtColor(img, (cv2.COLOR_RGB2YUV)) #converte l'immagine in YUV
    # ----- global equalization
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = (img / 255.).astype(np.float32) # rappresenta i valori in un range di [0-1]

    # ----- local equalization

    # An algorithm for local contrast enhancement, that uses histograms computed
    # 0 over different tile regions of the image. Local details can therefore be
    # enhanced even in regions that are darker or lighter than most of the image.
    img = (exposure.equalize_adapthist(img) - 0.5)

    return img

# ------------------------
# blurring
# ------------------------

def motion_blur(img):
    size = 4
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    img_bl = cv2.filter2D(img, -1, kernel_motion_blur)

    return img_bl


import numpy as np
import cv2
from skimage import exposure


def sharpen_img(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

def transform_img(img_in):
    img_in = img_in.copy()
    img_out = sharpen_img(img_in)
    img_out = cv2.cvtColor(img_in, cv2.COLOR_RGB2YUV)

    img_out[:,:,0] = cv2.equalizeHist(img_out[:,:,0])

    return img_out[:,:,0]

def random_rotate_img(img):
    c_x,c_y = int(img.shape[0]/2), int(img.shape[1]/2)
    ang = 30.0*np.random.rand()-15
    Mat = cv2.getRotationMatrix2D((c_x, c_y), ang, 1.0)
    return cv2.warpAffine(img, Mat, img.shape[:2])

def random_scale_img(img):
    img2=img.copy()
    sc_y=0.4*np.random.rand()+1.0
    img2=cv2.resize(img, None, fx=1, fy=sc_y, interpolation = cv2.INTER_CUBIC)

    dy = int((img2.shape[1]-img.shape[0])/2)
    end = img.shape[1]-dy
    img2 = img2[dy:end,:,:]
    assert img2.shape[0] == 32
#     print(img2.shape,dy,end)
    return img2

#Compute linear image transformation ing*s+m
def lin_img(img,s=1.0,m=0.0):
    img2=cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))

#Change image contrast; s>1 - increase
def contr_img(img, s=1.0):
    m = 127.0*(1.0-s)
    return lin_img(img, s, m)


def augment_img(img):
    img = img.copy()
    img = contr_img(img, 1.8*np.random.rand()+0.2)
    img = random_rotate_img(img)
    img = random_scale_img(img)

    # return transform_img(img)
    return normalize_img(img)




# -----------------------
# normalization
# -----------------------

def normalize_img(img):
    # img_bl = motion_blur(img)
    #img_bl = sharpen_img(img)
    img_y = cv2.cvtColor(img, (cv2.COLOR_BGR2YUV))[:,:,0]
    img_y = (img_y / 255.).astype(np.float32)

    # ----- use one of the follow
    # adjust_log : very very fast,
    # equalize_adapthist : really slow but accuracy gain
    #img_y = exposure.adjust_log(img_y)
    img_y = (exposure.equalize_adapthist(img_y) - 0.5)
    # -----
    img_y = img_y.reshape(img_y.shape + (1,))

    return img_y


# ------------------------
# blur
# ------------------------

def motion_blur(img):
    size = 4
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    img_bl = cv2.filter2D(img, -1, kernel_motion_blur)

    return img_bl

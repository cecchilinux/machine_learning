import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import my_mod_manipulate_image as mod

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
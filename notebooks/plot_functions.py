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
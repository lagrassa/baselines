from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model, load_model
from keras import backend as K

x_train = np.load("states.npy")/255.0

encoder = load_model("models/encoder.h5")
encoded = encoder.predict(x_train)

def plot_im_and_original(i):
    original = x_train[i]
    plt.imshow(original, cmap="gray")
    plt.show()
    flat = encoded[i].reshape((64,8))
    plt.imshow(flat, cmap="gray")
    plt.show()

plot_im_and_original(2)




#image and encoded version next to it: for interesting ones. 

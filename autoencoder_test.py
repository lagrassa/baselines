from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
from keras.models import Model, load_model
from keras import backend as K

x_train = np.load("states.npy")/255.0

encoder = load_model("models/encoder.h5")
import ipdb; ipdb.set_trace()
encoded = encoder.predict(x_train)
print(encoded.shape)

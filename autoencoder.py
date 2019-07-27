from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
from keras.models import Model, load_model
from keras import backend as K

input_img = Input(shape=(128, 128, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', name="firstone", padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding="same")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.callbacks import TensorBoard
data = np.load("states.npy")/255.0
split = int(data.shape[0]*0.85)
x_train = data[:split]
x_test = data[split:]

autoencoder.fit(x_train, x_train,
                validation_data =(x_test, x_test),
                epochs=150,
                batch_size=128,
                shuffle=True,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

encoder = Model(input_img, encoded)
encoder.save("models/encoder.h5")


#del encoder
#encoder = load_model("models/encoder.h5")
#encoder(x_train)

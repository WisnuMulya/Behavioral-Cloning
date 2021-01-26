# Import Packages
import tensorflow as tf
from keras.applications import InceptionV3
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Input, GlobalAveragePooling2D
from keras.models import Model

# Create the model
def build_model():
    """
    Build a model with pretrained AlexNet.
    """
    # Instantiate AlexNet
    inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(139,139,3))

    # Create preprocessing layers
    inp = Input(shape=(160, 320, 3))
    norm = Lambda(lambda x: (x/255)-0.5)(inp) # normalize image
    crop = Cropping2D(cropping=((50, 20), (0,0)))(norm)
    resized = Lambda(lambda x: tf.image.resize_images(x, (139, 139)))(crop)
    
    # Attach the AlexNet with the preprocessing layers
    piped_inception = inception(resized)
    
    # Attach new classifier layers
    avg_pool = GlobalAveragePooling2D()(piped_inception)
    fc = Dense(512, activation='relu')(avg_pool)
    prediction = Dense(1)(fc)
    model = Model(input=inp, output=prediction)

    return model

# Convert model
model = build_model()
model.compile(loss='mse', optimizer='adam')
model.load_weights('model.h5')
model.save('model_py37.h5') # convert to work on python==3.7

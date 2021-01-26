# Import Packages
import pandas as pd
import numpy as np
import os
from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage import io
import tensorflow as tf
from keras import Sequential
from keras.applications import InceptionV3
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.callback import ModelCheckpoint, EarlyStopping

# Load driving log into a dataframe
df = pd.read_csv('/opt/data/driving_log.csv', header=None,
                 names=['path_center', 'path_left', 'path_right', 'steering', 'throttle', 'brake', 'speed'])

# Samples generator
def sample_generator(samples, batch_size=32):
    """
    A dataset generator which takes a Pandas Dataframe as samples
    and outputs dataset in batches.
    INPUT:
    samples (Pandas DataFrame) - a dataframe of samples
    batch_size (int) - the size of one sample batch to be processed
    OUTPUT:
    X_train (numpy array) - a dataset of images
    y_train (numpy array) - a dataset of steerings
    """
    num_samples = len(samples)
    while True:
        samples = df.sample(frac=1, random_state=42) # shuffle samples in pandas dataframe

        # Create batches of dataset
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size].reset_index(drop=True)

            # Generate images 
            images = []
            steerings = []
            for idx, row in batch_samples.iterrows():
                # Add multiple camera data
                img_center_path = '/opt/data/IMG/' + row['path_center'].split('/')[-1]
                img_left_path = img_center_path.replace('center', 'left')
                img_right_path = img_center_path.replace('center', 'right')
                
                img_center = io.imread(img_center_path)
                img_left = io.imread(img_left_path)
                img_right = io.imread(img_right_path)
                center_steering = float(row['steering'])
                correction = 0.2
                left_steering = center_steering + correction
                right_steering = center_steering - correction
                
                # Augment data by flipping horizontally
                img_center_flipped = np.fliplr(img_center)
                img_left_flipped = np.fliplr(img_left)
                img_right_flipped = np.fliplr(img_right)
                center_steering_flipped = -center_steering
                left_steering_flipped = -left_steering
                right_steering_flipped = -right_steering

                # Append multiple camera data and their augmented version
                images.append(img_center)
                steerings.append(center_steering)
                images.append(img_center_flipped)
                steerings.append(center_steering_flip)
                images.append(img_left)
                steerings.append(left_steering)
                images.append(img_left_flipped)
                steerings.append(left_steering_flipped)
                images.append(img_right)
                steerings.append(right_steering)
                images.append(img_right_flipped)
                steerings.append(right_steering_flipped)

            # Transform batch dataset to a numpy array
            X_train = np.array(images)
            y_train = np.array(steerings)
            
            yield shuffle(X_train, y_train)

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

# Build samples generator
df = df.sample(frac=1, random_state=42) # shuffle the dataframe
train_samples, validation_samples = train_test_split(df, test_size=0.2)
train_generator = sample_generator(train_samples)
validation_generator = sample_generator(validation_samples)

# Create callbacks for training
save_path = 'model.h5'
checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)
stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=5)

# Train the model
batch_size = 32
model = build_model()
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator,
                    steps_per_epoch=ceil(len(train_samples)/batch_size),
                    validation_data=validation_generator,
                    validation_steps=ceil(len(validation_samples)/batch_size),
                    callbacks=[checkpoint, stopper],
                    epochs=1)

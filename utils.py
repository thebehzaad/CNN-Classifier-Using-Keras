import cv2
import random
import os
import numpy as np
from keras import layers
from keras import models
from keras import optimizers

def get_config():
    config={
    'nrows': 300,       # Height
    'ncolumns': 400,    # Width
    'nchannels': 3,     # Number of channels
    'training_batch_size': 32,
    'nepochs':64,           
    'rotation_range':40,
    'width_shift_range':0.2,
    'height_shift_range':0.2,
    'shear_range':0.2,
    'zoom_range':0.2,
    'horizontal_flip':True,
    'learning_rate':1e-4}
    return config


def train_read_and_process_image(train_path, config): 
    good_namelist = [train_path+'/good/{}'.format(i) for i in os.listdir(train_path+'/good/')]  
    flare_namelist = [train_path+'/flare/{}'.format(i) for i in os.listdir(train_path+'/flare/')]  
    X_image=[]
    y_label=[]
    for image in good_namelist:
        X_image.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (config['ncolumns'],config['nrows']), interpolation=cv2.INTER_AREA))  #Read the image
        y_label.append(0)
    for image in flare_namelist:
        X_image.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (config['ncolumns'],config['nrows']), interpolation=cv2.INTER_AREA))  #Read the image
        y_label.append(1)
    # Shuffling Images
    temp = list(zip(X_image, y_label))
    random.shuffle(temp)
    X_image, y_label = zip(*temp)
    X_image = np.array(X_image)
    y_label = np.array(y_label)
    return X_image,y_label


def test_read_and_process_image(test_path,config):
    test_namelist = [test_path+'/{}'.format(i) for i in os.listdir(test_path)]
    X_image=[]
    for image in test_namelist:
        X_image.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (config['ncolumns'],config['nrows']), interpolation=cv2.INTER_AREA))  #Read the image
    X_image = np.array(X_image)
    return X_image


def cnn_model(config):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(config['nrows'], config['ncolumns'], config['nchannels'])))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))  #Dropout for regularization
    model.add(layers.Dense(1, activation='sigmoid'))  
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=config['learning_rate']), metrics=['acc'])
    return model

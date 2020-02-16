"""*****************************************************************************************************

                                        CNN for Image Classification

*****************************************************************************************************"""
#%% Importing Necessary Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils import get_config, cnn_model, train_read_and_process_image, test_read_and_process_image

#%%

mode='Test'  # It can be either 'Training' or 'Test'
train_path='../training'  # the training path contains two subfolders of flare and good 
test_path='../test'       # the test path contains all the test images

#%%

config=get_config() #Hyperparameters

X_train, y_train= train_read_and_process_image(train_path, config) # Reading Training images
nsample=len(X_train)

# data augmentation generator
train_datagen = ImageDataGenerator(rescale=1./255,   # Scale the image between 0 and 1
                                    rotation_range=config['rotation_range'],
                                    width_shift_range=config['width_shift_range'],
                                    height_shift_range=config['height_shift_range'],
                                    shear_range=config['shear_range'],
                                    zoom_range=config['zoom_range'],
                                    horizontal_flip=config['horizontal_flip'])


#%% Training the Model

if mode=='Training':
    folds=StratifiedKFold(5).split(X_train, y_train)
    acc=[]
    val_acc=[]
    loss=[]
    val_loss=[]
    for j, (train_idx, val_idx) in enumerate(folds):    
        print('\nFold ',j)
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_val_cv = X_train[val_idx]
        y_val_cv= y_train[val_idx]
        ntrain=len(X_train_cv)
        model=cnn_model(config) # Buidling the CNN model
        #model.summary()
        train_generator = train_datagen.flow(X_train_cv, y_train_cv, batch_size=config['training_batch_size'])
        results=model.fit_generator(train_generator,
                                    steps_per_epoch=ntrain // config['training_batch_size'],
                                    epochs=config['nepochs'],
                                    validation_data=(X_val_cv*(1./255),y_val_cv))
        
        acc.append(results.history['acc'])
        val_acc.append(results.history['val_acc'])
        loss.append(results.history['loss'])
        val_loss.append(results.history['val_loss'])
    
    # Plotting the Average Performance for Different Epochs    
    avg_acc=np.mean(np.array(acc),axis=0)
    avg_val_acc=np.mean(np.array(val_acc),axis=0)
    avg_loss=np.mean(np.array(loss),axis=0)
    avg_val_loss=np.mean(np.array(val_loss),axis=0)
    epochs = range(1, len(avg_acc) + 1)    
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    ax[0].plot(epochs, avg_acc, 'b', label='Training accurarcy')
    ax[0].plot(epochs, avg_val_acc, 'r', label='Validation accurarcy')
    ax[0].set_title('Training and Validation accurarcy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epochs')
    ax[0].legend(loc='best')
    ax[1].plot(epochs, avg_loss, 'b', label='Training loss')
    ax[1].plot(epochs, avg_val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and Validation loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epochs')
    ax[1].legend(loc='best')
    plt.savefig('performance.png', dpi=300)
    plt.show()

elif mode=='Test':
    model=cnn_model(config)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=config['training_batch_size'])
    results=model.fit_generator(train_generator,
                                steps_per_epoch=nsample // config['training_batch_size'],
                                epochs=config['nepochs'])

#Saveing the model
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

#%% Test Labels

if mode=='Test':
    X_test= test_read_and_process_image(test_path,config)
    X_test=X_test*(1./255)
    pred = model.predict(X_test).reshape(-1)
    pred=pred.astype(int)
    for i in range(len(pred)):
        print('Label is {}'.format(pred[i]))

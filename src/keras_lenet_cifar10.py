# -*- coding: UTF-8 -*-
"""Keras LeNet-5 with Cifar10

It Implements LeNet-5 in Keras, and applied on cifar10 dataset

Usage: 
    python3 keras_lenet_cifar10.py 

Author: Anto Tu
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

"""
def normalize(x_train, x_test):
        mean = np.mean(x_train, axis=(0,1,2,3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train-mean) / (std+1e-7)
        x_test = (x_test-mean) / (std+1e-7)
        return x_train, X_test
"""

def prepare_cifar10_dataset():
    # get cifar10 training and test data
    (in_train, out_train), (in_test, out_test) = cifar10.load_data()

    """
    ## Normalize Training and Testset
    x_train, x_test = normalize(x_train, x_test)

    ## OneHot Label 由(None, 1)-(None, 10)
    one_hot=OneHotEncoder()
    y_train=one_hot.fit_transform(y_train).toarray()
    y_test=one_hot.transform(y_test).toarray()
    """

class LeNet5():
    #staticmethod
    def build_model(input_shape, classes):
        """build Lenet-5 keras model
        
        Args:
            input_shape:
            class_num: number of result class

        Return: model
        """
        model = Sequential()

        # 1st CNN / MaxPooling
        model.add(Conv2D(32, kernel_size=(3, 3), 
                       input_shape=(width, width, depth), 
                       activation='relu'))
        #model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))

        # 2nd CNN / MaxPooling
        model.add(Conv2D(32, kernel_size(3, 3), activation='relu'))
        #model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))

        # flatten
        model.add(Flatten())

        # full connection 
        model.add(Dense(output_dim=100,activation='relu'))
        model.add(Dropout(p=0.3))
        model.add(Dense(output_dim=classes,activation='softmax'))

        # build model
        model.compile(optimizer = 'adam', 
                           loss = 'categorical_crossentropy', 
                           metrics = ['accuracy'])

        return model


def main():
    """main function"""
    
    (in_train, out_train, in_test, out_test) = prepare_cifar10_dataset():
    model = LeNet5.build_model(input_shape=(32, 32, 3), classes=10)
    model.fit(in_train, out_train, batch_size=100, epochs=3)
    history = model.fit(in_test/255, to_categorical(out_test), batch_size=32)


if __name__ == '__name__':
    main()    


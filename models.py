from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
import numpy as np

# MUSIC CNN

def create_cnn(height,width,  depth, filters=(16,16,16,32,32), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1
    # define the model input
    inputs = Input(shape=inputShape)
    # loop over the number of filters
    #
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs


        # CONV => RELU => BN => POOL
        x= Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)


    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    # x1= Dropout(0.5)(x)

    # # apply another FC layer, this one to match the number of nodes
    # # coming out of the MLP
    # x1 = Dense(4)(x1)
    # x1 = Activation("relu")(x1)
    # x2 = Dense(4)(x2)
    # x2 = Activation("relu")(x2)

    # check to see if the regression node should be added
    if regress:
        y1 = Dense(1, activation="linear")(x)
        y2 = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, outputs=[y1, y2])
    #
    return model

    #for one output
    #
    # for (i, f) in enumerate(filters):
    #     # if this is the first CONV layer then set the input
    #     # appropriately
    #     print('dadaist')
    #     if i == 0:
    #         x = inputs
    #
    #     # CONV => RELU => BN => POOL
    #     x = Conv2D(f, (3, 3), padding="same")(x)
    #     x = Activation("relu")(x)
    #     x = BatchNormalization(axis=chanDim)(x)
    #
    #     # flatten the volume, then FC => RELU => BN => DROPOUT
    # x = Flatten()(x)
    # x = Dense(16)(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(axis=chanDim)(x)
    # x = Dropout(0.5)(x)
    #
    # # apply another FC layer, this one to match the number of nodes
    # # coming out of the MLP
    # x = Dense(4)(x)
    # x = Activation("relu")(x)
    #
    # # check to see if the regression node should be added
    # if regress:
    #     x = Dense(1, activation="linear")(x)
    #
    # # construct the CNN
    # model = Model(inputs, x)
    # #






# def create_cnn(height,width, depth, filters=(16,32,64), regress=False):
#     # initialize the input shape and channel dimension, assuming
#     # TensorFlow/channels-last ordering
#     inputShape = (height, width, depth)
#     chanDim = -1
#     # define the model input
#     inputs = Input(shape=inputShape)
#     # loop over the number of filters
#     for (i, f) in enumerate(filters):
#         # if this is the first CONV layer then set the input
#         # appropriately
#         print('dadaist')
#         if i == 0:
#             x = inputs
#
#         # CONV => RELU => BN => POOL
#         x1 = Conv2D(f, (3, 3), padding="same")(x)
#         x = Activation("relu")(x)
#         x = BatchNormalization(axis=chanDim)(x)
#         # x = MaxPooling2D(pool_size=(2, 2))(x)
#
#     # flatten the volume, then FC => RELU => BN => DROPOUT
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Flatten()(x)
#     x = Dense(64)(x)
#     x = Activation("relu")(x)
#     x = BatchNormalization(axis=chanDim)(x)
#     x = Dense(64)(x)
#     x = Activation("relu")(x)
#     x = BatchNormalization(axis=chanDim)(x)
#     # check to see if the regression node should be added
#     if regress:
#         y1 = Dense(1, activation="linear")(x)
#         y2 = Dense(1, activation="linear")(x)
#     # construct the CNN
#     model = Model(inputs, outputs=[y1, y2])
#
#
#
#     return model


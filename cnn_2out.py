# -*- coding: utf-8 -*-
import argparse
import pathlib
import scipy.io as scio
from sklearn.model_selection import train_test_split
import models as models
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import locale
import keras
from keras import optimizers
from keras.utils import plot_model
skip_training = False
contract_background = False
from keras.utils import plot_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'please enter the input and output folder '
    parser.add_argument('INPUT', help='Path to recorded dataset directory')
    args = vars(parser.parse_args())
    in_root = pathlib.Path(args['INPUT']).resolve()
    i = 0
    if contract_background:
        background = scio.loadmat("background.mat")
        background = background['background']
        background = np.reshape(background,(64,64))
        print(background.shape)
    for file in os.listdir(in_root):
        dataFile = str(in_root.joinpath(file))
        data = scio.loadmat(dataFile)
        if i==0:

            # fft = data['new_fft']
            music = data['new_music']
            position = data['position']
        else:
            # fft = np.concatenate((fft, data['new_fft']))
            music = np.concatenate((music, data['new_music']))
            position = np.concatenate((position, data['position']))

        i=i+1
    if contract_background:
        music = music-background

    ##delete outlier
    i=0
    while i < music.shape[0]:
       # if position[i, 2] < 1:
       if position[i, 2] < 1.5  :

            music = np.delete(music,i,axis=0)
            position = np.delete(position, i, axis=0)
       else:
           i=i+1

    # while i < music.shape[0]:
    #     # if position[i, 2] < 1:
    #     if position[i, 2] > 4.76:
    #
    #         music = np.delete(music, i, axis=0)
    #         position = np.delete(position, i, axis=0)
    #     else:
    #         i = i + 1

    i = 0
    while i < music.shape[0]:
        if  np.isnan(position[i, 2]):

            music = np.delete(music, i, axis=0)
            position = np.delete(position, i, axis=0)
        else:
            i = i + 1




    print(music.shape)
    print(position.shape)


    # print(type(fft))
    # position = position[:,2]
    # music_max = music.max()
    # music = music/music_max
    # position_max = position[:,2].max()
    # position = (position[:,2]/position_max)
    (trainX, splitX, trainY, spitY)  = train_test_split(music,position,test_size=0.3, random_state=42)
    (valX, testX, valY, testY) = train_test_split(music,position,test_size=0.25, random_state=42)
    # print(np.shape(trainX))
    # trainX = trainX.reshape(np.shape(trainX)[0], np.shape(trainX)[1],np.shape(trainX)[2],1)
    # maxrange = position[:,2].max()
    # trainY = trainY / maxrange
    # testY = testY / maxrange
    # print(np.shape(trainX))
    if not skip_training:
        model = models.create_cnn(64, 64, 1, regress=True)
        sgd = optimizers.SGD(lr=2e-4, decay=1e-6, momentum=0.9)
        opt = Adam(lr=1e-4, decay=1e-3 / 200)
        print(model.summary())
        model.compile(loss=[keras.losses.MeanSquaredError(),
                        keras.losses.MeanSquaredError(),
                        ], optimizer=sgd,loss_weights=[1,1])
        # callback = keras.callbacks.EarlyStopping(monitor="val_loss",patience=)
        # train the model
        print("[INFO] training model...")
        keras.utils.plot_model(model, to_file='model_1.png', show_shapes=True)
        history = model.fit(trainX, [trainY[:,0],trainY[:,2]], validation_data=(valX, [valY[:,0], valY[:,2]]),
                  epochs= 1500,batch_size=256)

        print("[INFO] predicting ...")
        preds = model.predict(testX)
        print(model.summary())
    else:
        model = keras.models.load_model("MyModel")
        print(model.summary())
        preds = model.predict(testX)

    # scores = model.evaluate(testX, testY, verbose=0)



    y0 = preds[0]
    y1 = preds[1]
    # print(preds)
    y0 = np.reshape(y0, (1, -1))
    TestY0 = np.reshape(testY[:, 0], (1, -1))
    diff0 = abs(y0 - TestY0)
    print(np.average(diff0))

    y1 = np.reshape(y1, (1, -1))
    TestY1 = np.reshape(testY[:, 2], (1, -1))
    diff1 = abs(y1 - TestY1)
    print(np.average(diff1))

    diff_squre1 = np.power(diff0, 2)
    diff_squre2 = np.power(diff1, 2)
    diff_euclidean = np.sqrt(diff_squre1 + diff_squre2)
    print(np.average(diff_euclidean))

    a = np.arange(len(TestY1))
    # x = np.arange(1.5, 4.7, 0.01)  # print(np.shape(a))
    x = np.arange(-1.5, 4, 0.01)
    y = x
    if not skip_training:
        model.save('MyModel')
        plt.xlim((1, 4.5))
        plt.ylim((1, 4.5))
        plt.scatter(y1, TestY1, c="r", alpha=0.5, label="preds_zs")
        plt.plot(x, y)
        plt.legend(loc="best")
        plt.show()

        plt.xlim((-1, 1))
        plt.ylim((-1, 1))

        plt.scatter(y0, TestY0, c="g", alpha=0.5, label="preds_xs")
        plt.plot(x, y)
        plt.legend(loc="best")
        plt.show()

        plt.scatter(y0, y1, c="r", alpha=0.5)
        plt.scatter(TestY0, TestY1, c="g", alpha=0.5)
        plt.show()

        plt.scatter(y0,y1,c="r",alpha=0.5)
        plt.scatter(TestY0, TestY1, c="g", alpha=0.5)
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    else:
        # plt.xlim((-1, 1.5))
        # #
        # plt.scatter(y0, y1, c="r", alpha=0.5)
        # plt.scatter(TestY0, TestY1, c="g", alpha=0.5)
        # plt.xlim((1.5, 4.7))
        # plt.ylim((1.5, 4.7))
        plt.xlim((1, 4.5))
        plt.ylim((1, 4.5))
        plt.scatter(y1, TestY1, c="r", alpha=0.5, label="preds_zs")
        plt.plot(x, y)
        plt.legend(loc="best")
        plt.show()

        plt.xlim((-1, 1))
        plt.ylim((-1, 1))

        plt.scatter(y0, TestY0, c="g", alpha=0.5, label="preds_xs")
        plt.plot(x, y)
        plt.legend(loc="best")
        plt.show()

        plt.scatter(y0, y1, c="r", alpha=0.5)
        plt.scatter(TestY0, TestY1, c="g", alpha=0.5)
        plt.show()
    # print(scores)

        # # dataNew = str(in_root.joinpath('result.mat'))
        # # scio.savemat(dataNew, {'preds': preds, 'ys': testY})
        #
        #

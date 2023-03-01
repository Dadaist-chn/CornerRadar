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
from keras.utils import plot_model
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import pykalman
from matplotlib.pyplot import MultipleLocator
import matplotlib.font_manager as font_manager
from pykalman import KalmanFilter
import time

contract_background = False


fig, ax = plt.subplots()

xdata, ydata = [], []
ln, = plt.plot([], [], 'go')

sample =True
def init():
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 6)
    return ln


def update(n):
    x = position[n][0]
    y = position[n][2]
    xdata.append(x)
    ydata.append(y)
    ln.set_data(xdata, ydata)

    return ln


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'please enter the input and output folder '
    parser.add_argument('INPUT', help='Path to recorded dataset directory')
    args = vars(parser.parse_args())
    in_root = pathlib.Path(args['INPUT']).resolve()
    i = 0
    sample = True

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

    ##delete outlier
    i=0
    while i < music.shape[0]:
       # if position[i, 2] < 1:
       if position[i, 2] < 1.5  :

            music = np.delete(music,i,axis=0)
            position = np.delete(position, i, axis=0)
       else:
           i=i+1

    i = 0
    while i < music.shape[0]:
        if  np.isnan(position[i, 2]):

            music = np.delete(music, i, axis=0)
            position = np.delete(position, i, axis=0)
        else:
            i = i + 1

    if contract_background:
        music = music - background
    testX = music
    testY = position
    print(music.shape)
    print(position.shape)
    model = keras.models.load_model("MyModel")
    keras.utils.plot_model(model, to_file='model_1.png', show_shapes=True)
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

    plt.xlim((-0.75, 1))
    plt.ylim((1.3, 4.2))
    plt.scatter(y0, y1, c="r", alpha=0.5, label="predicted")
    plt.scatter(TestY0, TestY1, c="g", alpha=0.5, label="ground truth")
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size=20)
    plt.legend(loc='lower right',prop=font)
    ax = plt.gca()
    ax.set_aspect(1)
    # ani = animation.FuncAnimation(fig, update, np.arange(1, 1000), init_func=init, interval=1000, blit=False)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fm'))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2fm'))
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)

    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    # 保存动图
    # ani.save('scatter.html', writer='imagemagick', fps=5)
    plt.show()


    #kalman filter
    measurements = np.c_[y0.T,y1.T]
    dt = 1/30  # time step
    transition_matrix = [[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]]

    # Define the observation matrix
    # We assume that we can directly observe the position (x, y) of the object
    observation_matrix = [[1, 0, 0, 0],
                          [0, 1, 0, 0]]

    # Define the initial state
    # We assume that the object starts at the origin with zero velocity
    initial_state_mean = [0, 0, 0, 0]
    initial_state_covariance = np.eye(4)

    # Define the process noise covariance
    # We assume that the velocity of the object has some random noise
    # but the position is measured directly
    # So the process noise covariance is diagonal with large values for the velocities
    # and small values for the positions
    process_noise_covariance = np.diag([0.1, 0.1, 1.0, 1.0])

    # Define the measurement noise covariance
    # We assume that the measurement noise is small and isotropic
    measurement_noise_covariance = np.eye(2) * 0.1

    # Create the Kalman filter
    kf = pykalman.KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        transition_covariance=process_noise_covariance,
        observation_covariance=measurement_noise_covariance
    )

    # Generate some example data
    # We assume that the true location of the object is a sinusoidal curve

    # Use the Kalman filter to estimate the true location of the object
    filtered_states, _ = kf.filter(measurements)

    # Print the estimated location of the object
    print(filtered_states[:, :2])


    y0 = filtered_states[:, 0]
    y1 = filtered_states[:, 1]
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

    plt.xlim((-0.75, 1))
    plt.ylim((1.3, 4.2))
    plt.scatter(y0, y1, c="r", alpha=0.5, label="kalman")
    plt.scatter(preds[0], preds[1], c="g", alpha=0.5, label="predicted")
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size=20)
    plt.legend(loc='lower right', prop=font)
    ax = plt.gca()
    ax.set_aspect(1)
    # ani = animation.FuncAnimation(fig, update, np.arange(1, 1000), init_func=init, interval=1000, blit=False)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fm'))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2fm'))
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)

    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    # 保存动图
    # ani.save('scatter.html', writer='imagemagick', fps=5)
    plt.show()

    y0 = filtered_states[:, 0]
    y1 = filtered_states[:, 1]
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

    plt.xlim((-0.75, 1))
    plt.ylim((1.3, 4.2))
    plt.scatter(y0, y1, c="r", alpha=0.5, label="Kalman")
    plt.scatter(TestY0, TestY1, c="g", alpha=0.5, label="ground truth")
    font = font_manager.FontProperties(family='Times New Roman',
                                       weight='bold',
                                       style='normal', size=20)
    plt.legend(loc='lower right', prop=font)
    ax = plt.gca()
    ax.set_aspect(1)
    # ani = animation.FuncAnimation(fig, update, np.arange(1, 1000), init_func=init, interval=1000, blit=False)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fm'))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2fm'))
    plt.yticks(fontproperties='Times New Roman', size=25)
    plt.xticks(fontproperties='Times New Roman', size=25)

    ax.xaxis.set_major_locator(MultipleLocator(0.4))
    # 保存动图
    # ani.save('scatter.html', writer='imagemagick', fps=5)
    plt.show()
        # # dataNew = str(in_root.joinpath('result.mat'))
        # # scio.savemat(dataNew, {'preds': preds, 'ys': testY})
        #
        #


    ## results from different distance
    close = []
    middle = []
    far = []
    # print(y1.shape)
    y0=y0.T
    y1=y1.T
    TestY0=TestY0.T
    TestY1=TestY1.T
    for i in range(len(y0)):
        if y1[i] < 2.3:
            close.append([y0[i],y1[i],TestY0[i],TestY1[i]])
        elif y1[i] < 3.1:
            middle.append([y0[i], y1[i], TestY0[i], TestY1[i]])
        elif y1[i] < 4.0:
            far.append([y0[i], y1[i], TestY0[i], TestY1[i]])

    close = np.array(close)
    close=np.reshape(close,(close.shape[0],close.shape[1]))
    middle = np.array(middle)
    middle = np.reshape(middle, (middle.shape[0],middle.shape[1]))
    far= np.array(far)
    far = np.reshape(far, (far.shape[0],far.shape[1]))
    print(close.shape)
    print(middle.shape)
    print(far.shape)

    close_x_mae=0
    close_y_mae = 0
    middle_x_mae=0
    middle_y_mae = 0
    far_x_mae=0
    far_y_mae = 0

    for i in range(close.shape[0]):
        close_x_mae = close_x_mae + abs(close[i,0] - close[i,2])
        close_y_mae = close_y_mae + abs(close[i,1] - close[i,3])
    close_x_mae=close_x_mae/close.shape[0]
    close_y_mae=close_y_mae/close.shape[0]

    print(close_x_mae)
    print(close_y_mae)

    for i in range(middle.shape[0]):
        middle_x_mae = middle_x_mae + abs(middle[i,0] - middle[i,2])
        middle_y_mae = middle_y_mae + abs(middle[i,1] - middle[i,3])
    middle_x_mae=middle_x_mae/middle.shape[0]
    middle_y_mae=middle_y_mae/middle.shape[0]

    print(middle_x_mae)
    print(middle_y_mae)

    for i in range(far.shape[0]):
        far_x_mae = far_x_mae + abs(far[i, 0] - far[i, 2])
        far_y_mae = far_y_mae + abs(far[i, 1] - far[i, 3])
    far_x_mae = far_x_mae / far.shape[0]
    far_y_mae = far_y_mae / far.shape[0]

    print(far_x_mae)
    print(far_y_mae)
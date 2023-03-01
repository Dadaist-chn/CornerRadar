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

    obs=np.c_[y0.T,y1.T]
    #kalman filter
    # measurements = np.c_[y0.T,y1.T]
    # print(measurements.shape)
    # initial_state_mean = [measurements[0, 0],
    #                       0,
    #                       measurements[0, 1],
    #                       0]
    #
    # transition_matrix = [[1, 1, 0, 0],
    #                      [0, 1, 0, 0],
    #                      [0, 0, 1, 1],
    #                      [0, 0, 0, 1]]
    #
    # observation_matrix = [[1, 0, 0, 0],
    #                       [0, 0, 1, 0]]
    #
    # kf1 = KalmanFilter(transition_matrices=transition_matrix,
    #                    observation_matrices=observation_matrix,
    #                    initial_state_mean=initial_state_mean
    #                    )
    #
    # kf1 = kf1.em(measurements, n_iter=5)
    # (filtered_state_means, filtered_state_covariances) = kf1.filter(measurements)
    # (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(np.c_[filtered_state_means[:, 0],filtered_state_means[:, 2]])
    #



  # particle filter
    def particle_filter(obs, n_particles=1000, n_steps=10, sigma=0.1):
        """
        Implementation of a particle filter for filtering location data.


        Parameters:
        obs (numpy array): An array of shape (n, 2) representing the observed location data.
        n_particles (int): The number of particles to use in the filter.
        n_steps (int): The number of time steps to run the filter for.
        sigma (float): The standard deviation of the noise added to the particle positions at each time step.

        Returns:
        numpy array: An array of shape (n_steps, 2) representing the filtered location data.
        """

        # Initialize particles with random positions and weights
        particles = np.random.rand(n_particles, 2)
        weights = np.ones(n_particles) / n_particles

        # Loop over time steps
        filtered = np.zeros((n_steps, 2))
        for i in range(n_steps):
            # Resample particles according to their weights
            idx = np.random.choice(np.arange(n_particles), size=n_particles, replace=True, p=weights)
            particles = particles[idx]

            # Add noise to particles
            particles += np.random.normal(loc=0, scale=sigma, size=(n_particles, 2))

            # Calculate particle weights based on how well they explain the observation
            dists = np.linalg.norm(particles - obs[i], axis=1)
            weights = np.exp(-dists ** 2 / (2 * sigma ** 2))

            # Normalize weights
            weights /= np.sum(weights)

            # Calculate filtered location as the weighted average of particles
            filtered[i] = np.average(particles, weights=weights, axis=0)

        return filtered



    # (smoothed_state_means, smoothed_state_covariances) = kf1.filter(measurements)

    # times = range(measurements.shape[0])
    # plt.plot(times, measurements[:, 0], 'bo',
    #          times, measurements[:, 1], 'ro',
    #          times, smoothed_state_means[:, 0], 'b--',
    #          times, smoothed_state_means[:, 2], 'r--', )
    # plt.show()
    print(obs.shape)
    filtered = particle_filter(obs, n_particles=1000, n_steps=1049, sigma=0.1)
    print(filtered)
    # print(preds)



    y0 = filtered[:, 0]
    y1 = filtered[:, 1]
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
    plt.scatter(y0, y1, c="r", alpha=0.5, label="Particle filter")
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
    #     # # dataNew = str(in_root.joinpath('result.mat'))
    #     # # scio.savemat(dataNew, {'preds': preds, 'ys': testY})
    #     #
    #     #

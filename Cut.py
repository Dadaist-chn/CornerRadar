# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import pathlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio
import pandas as pd



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'please enter the input folder '
    parser.add_argument('INPUT', help='Path to recorded dataset directory')

    args = vars(parser.parse_args())

    # if not pathlib.Path(args['INPUT']).is_dir():''
    #     raise FileNotFoundError(args['INPUT'])

    in_root = pathlib.Path(args['INPUT']).resolve()
    dataFile = str(in_root.joinpath('radar.mat'))
    txtFile = str(in_root.joinpath('Target_position.txt'))
    excelFile = str(in_root.joinpath('timestamps.csv'))
    # with
    # start_percentage = float(args['S_P'])
    # end_percentage = float(args['E_P'])

    data = scio.loadmat(dataFile)


    # fft= data['fft']
    music = data['music']
    frame_num = music.shape[0]




    in_root = pathlib.Path(args['INPUT']).resolve()
    with open(str(in_root.joinpath('Target_position.txt')), "r") as f:  # 打开文件
        file = f.read()  # 读取文件
        ind = file.index("position:")
        newlist = file.split("\n")
        end = len(newlist[0])
        for i in range(len(newlist)):
            newlist[i] = newlist[i][ind + 9:end]
            newlist[i] = newlist[i].split(',')
        newlist = newlist[0:len(newlist) - 1]
        flatten_list = [element for sublist in newlist for element in sublist]
        newarray = np.array(flatten_list)
        newarray = map(float, newarray)
        newarray = np.array(newarray)
        newarray = newarray.reshape(-1, 3)

    with open(str(in_root.joinpath('Target_position.txt')), "r") as f:  # 打开文件
        file = f.read()  # 读取文件
        newlist = file.split("\n")
        end = len(newlist[0])
        for i in range(len(newlist)):
            newlist[i] = newlist[i][11:26]
        newlist = newlist[0:len(newlist) - 1]
        first_time = int(newlist[0][3:5])*1000*60 + int(newlist[0][6:8])*1000 + int(newlist[0][12:15])
        print('frist time:'+str(first_time))
        end_time = int(newlist[len(newlist) - 1][3:5]) * 1000 * 60 + int(newlist[len(newlist) - 1][6:8]) * 1000 + int(newlist[len(newlist) - 1][12:15])
        print('end_time:' + str(end_time))
        for i in range(len(newlist)):
            min = int(newlist[i][3:5])
            second = int(newlist[i][6:8])
            ms = int(newlist[i][12:15])
            now_time = min*60*1000 + second*1000 + ms - first_time
            newlist[i]=now_time
        newlist = np.array(newlist)
        newlist = newlist.reshape(-1,1)
        print(newlist.shape)

    with open(str(in_root.joinpath('timestamps.csv')), "r") as f:  # 打开文件
        file = f.read()  # 读取文件
        file = file.split("\n")
        start = (file[0][20:32])
        end = file[2][18:30]
        start_time_radar = int(start[0:2])*1000*60+int(start[3:5])*1000+int(start[6:9])
        end_time_radar = int(end[0:2])*1000*60+int(end[3:5])*1000+int(end[6:9])
        print('start_time_radar '+str(start_time_radar))
        print('end_time_radar ' + str(end_time_radar ))
        duration = end_time_radar - start_time_radar
        print('duration '+ str(duration))
        start = first_time - start_time_radar
        end = end_time - start_time_radar
        start_percentage = float(start)/duration * 100
        end_percentage = float(end)/duration * 100
        print("start_percentage:"+str(start_percentage))
        print("end_percentage:" + str(end_percentage))

    start_frame = int(frame_num * start_percentage * 0.01)
    end_frame = int(frame_num * end_percentage * 0.01)
    # new_fft = fft[start_frame-1 : end_frame-1,:,:]
    new_music = music[start_frame - 1: end_frame - 1, :, :]
    frame_num = new_music.shape[0]  # cut the part without video frame
    max_time = newlist[len(newlist)-1]
    newlist = newlist*(frame_num-1)/max_time
    newarray = np.c_[newarray,newlist]
    j=0
    duplicate = -1
    if (newarray[0][3] == newarray[1][3]):
        duplicate = 1
        newarray = np.delete(newarray, duplicate, axis=0)
        duplicate=-1
    for i in range(newarray.shape[0]):
        if (i!=0)&(newarray[i][3] == newarray[i-1][3]):
            duplicate=i
    if duplicate>0:
        newarray = np.delete(newarray, duplicate, axis=0)

    duplicate = -1
    for i in range(newarray.shape[0]):
        if (i!=0)&(newarray[i][3] == newarray[i-1][3]):
            duplicate=i
    if duplicate>0:
        newarray = np.delete(newarray, duplicate, axis=0)

    duplicate = -1
    for i in range(newarray.shape[0]):
        if (i!=0)&(newarray[i][3] == newarray[i-1][3]):
            duplicate=i
    if duplicate>0:
        newarray = np.delete(newarray, duplicate, axis=0)

    duplicate = -1
    for i in range(newarray.shape[0]):
        if (i!=0)&(newarray[i][3] == newarray[i-1][3]):
            duplicate=i
    if duplicate>0:
        newarray = np.delete(newarray, duplicate, axis=0)





    for i in range(frame_num):
        if i == newarray[j][3]:
            j=j+1
        else:
            a = [np.nan,np.nan,np.nan,i]
            newarray = np.insert(newarray, i, a, axis=0)
            j=j+1
    data1 = pd.Series(newarray[:,0])
    data1.interpolate(method='linear', limit=50,inplace=True)
    newarray[:,0]=data1
    data2 = pd.Series(newarray[:, 1])
    data2.interpolate(method='linear', limit=50, inplace=True)
    newarray[:, 1] = data1
    data3 = pd.Series(newarray[:, 2])
    data3.interpolate(method='linear', limit=50, inplace=True)
    newarray[:, 2] = data3
    # dataNew = str(in_root.joinpath('cut_radar.mat'))
    dataNew = str(in_root.joinpath('cut_' + args['INPUT'] + '.mat'))
    scio.savemat(dataNew, {'new_music': new_music,'position':newarray})





#! /usr/bin/env python3
import os
import sys
import time
import numpy as np
import math
import rospy
import matplotlib.pyplot as plt
import scipy.io as io
import torch

from cdpr import CDPR
from transform import euler2quaternion
from Jacobian import getJacobian, get_K_Jacobian
from mpc import MPController
from learn.models.model_general_nn import GeneralNN
from trainer import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_order = 'collision04'
X_data_save_path = 'traject_data/0518/' + file_order + '_X.txt'
U_data_save_path = 'traject_data/0518/' + file_order + '_U.txt'
dX_data_save_path = 'traject_data/0518/' + file_order + '_dX.txt'


'''
X: position
U: Motor Velocity
dX: position difference
'''


if __name__=="__main__":

    cdpr = CDPR()

    T = 0.1     # control period
    rate = rospy.Rate(1/T)
    
    x_r_list, y_r_list, z_r_list = [], [], []
    x_list, y_list, z_list = [], [], []

    X_data_list = np.empty((0, 3))
    U_data_list = np.empty((0, 3))
    dX_data_list = np.empty((0, 3))

    x0, y0, z0, _ = cdpr.getMovingPlatformPose()
    last_X = np.array([x0, y0, z0])


    #############################################
    #               Model Training
    #############################################

    # file_order = '01'
    # X_path = 'origin_data/random/' + file_order + '_X.txt'
    # U_path = 'origin_data/random/' + file_order + '_U.txt'
    # dX_path = 'origin_data/random/' + file_order + '_dX.txt'
    # X_path = 'sim_origin_data/5hz/' + file_order + '_X.txt'
    # U_path = 'sim_origin_data/5hz/' + file_order + '_U.txt'
    # dX_path = 'sim_origin_data/5hz/' + file_order + '_dX.txt'
    # X_path = 'traject_data/0508/' + file_order + '_X.txt'
    # U_path = 'traject_data/0508/' + file_order + '_U.txt'
    # dX_path = 'traject_data/0508/' + file_order + '_dX.txt'

    file_order = '1'
    X_path = 'origin_data/random_collision/' + file_order + '_X.txt'
    U_path = 'origin_data/random_collision/' + file_order + '_U.txt'
    dX_path = 'origin_data/random_collision/' + file_order + '_dX.txt'

    X_data_list = np.loadtxt(X_path, dtype='float')
    U_data_list = np.loadtxt(U_path, dtype='float')
    dX_data_list = np.loadtxt(dX_path, dtype='float') * T

    for i in range(2, 12):
        file_order = '{}'.format(i)
        X_path = 'origin_data/random_collision/' + file_order + '_X.txt'
        U_path = 'origin_data/random_collision/' + file_order + '_U.txt'
        dX_path = 'origin_data/random_collision/' + file_order + '_dX.txt'

        X_new = np.loadtxt(X_path, dtype='float')
        U_new = np.loadtxt(U_path, dtype='float')
        dX_new = np.loadtxt(dX_path, dtype='float') * T

        X_data_list = np.append(X_data_list, X_new, axis = 0)
        U_data_list = np.append(U_data_list, U_new, axis = 0)
        dX_data_list = np.append(dX_data_list, dX_new, axis = 0)

    print(np.max(U_data_list))
    model, train_log = train_model(X_data_list, U_data_list, dX_data_list)
    # save model and train log
    torch.save(model.state_dict(), 'model.pth')
    np.save('train_log.npy', train_log)
    # plot loss
    train_log = np.load('train_log.npy', allow_pickle=True).item()
    train_loss = train_log['trainerror']
    test_loss = train_log['testerror']
    fig = plt.figure(1)
    loss_plot = fig.add_subplot(1,1,1)
    plt.ion()
    loss_plot.plot(np.arange(0, len(train_loss)), train_loss, label='Training Loss')
    loss_plot.plot(np.arange(0, len(test_loss)), test_loss, label='Validation Loss')
    loss_plot.set_xlabel('epoches')
    loss_plot.set_ylabel('loss')
    loss_plot.legend()

    plt.ioff()
    # plt.show()

    # load model
    model = GeneralNN().to(device)
    state_dict=torch.load('model.pth')
    model.load_state_dict(state_dict)
    model.half()
    model.cuda()
    model.eval()
    model.preprocess_cuda((X_data_list.squeeze(), U_data_list, dX_data_list.squeeze()))

    # initialize MPC controller
    mpc_horizon = 5
    mpc = MPController(model, horizon=mpc_horizon)


    #############################################
    #               Main Loop
    #############################################

    ctrlCnt = 0

    cdpr.pretighten()

    while not rospy.is_shutdown() and ctrlCnt < 40 / T:

        if not cdpr.check_in_workspace():
            # pass
            print('SURPASS')
            cdpr.setMotorVelo(0, 0, 0)
            break

        print('------------------------------------------------------------------------')
        print('run: {}'.format(ctrlCnt))
        start_time = time.time()
        
        runTime = T * ctrlCnt

        x_ref_seq = np.empty(0)
        y_ref_seq = np.empty(0)
        z_ref_seq = np.empty(0)
        
        # referrence pose of moment k  (unit: degree)
        t = ctrlCnt * T
        for i in range(mpc_horizon):
            x_ref_seq = np.append(x_ref_seq, 0.08 * math.sin(math.pi/20 * (t + i * T)) + cdpr.xOff)
            y_ref_seq = np.append(y_ref_seq, 0.08 * math.cos(math.pi/20 * (t + i * T)) + cdpr.yOff)
            z_ref_seq = np.append(z_ref_seq, 0.05 * math.sin(math.pi/10 * (t + i * T)) + cdpr.zOff)
        print('target(position): {}'.format([x_ref_seq[0], y_ref_seq[0], z_ref_seq[0]]))


        # pose of moment k  (unit: degree)
        x, y, z, orientation = cdpr.getMovingPlatformPose()
        X = np.array([x, y, z])
        print('state(position): {}'.format(X))


        #############################################
        #               MPC Controller
        #############################################
        
        mpc_time = time.time()
        motor_velo = mpc.get_action(x_ref_seq, y_ref_seq, z_ref_seq, X)
        print('mpc time: {}'.format(time.time() - mpc_time))
        print("action(motor velo): {}".format(motor_velo))
        

        #############################################
        #                   sample
        #############################################

        # get the pose of the end effector
        x0, y0, z0, _ = cdpr.getMovingPlatformPose()
        X = np.array([x0, y0, z0])

        # set cable velocity in joint space
        for i in range(3):
            if np.abs(motor_velo[i]) > 350:
                motor_velo[i] = 350 * np.sign(motor_velo[i])
        cdpr.setMotorVelo(int(motor_velo[0]), int(motor_velo[1]), int(motor_velo[2]))

        
        # generate input and output data (except for the first run)
        if ctrlCnt > 0:
            X_data = last_X
            U_data = np.array([int(motor_velo[0]), int(motor_velo[1]), int(motor_velo[2])])
            dX_data = (X - last_X) / T      # convert to velocity

            X_data_list = np.append(X_data_list, X_data.reshape(1, 3), axis = 0)
            U_data_list = np.append(U_data_list, U_data.reshape(1, 3), axis = 0)
            dX_data_list = np.append(dX_data_list, dX_data.reshape(1, 3), axis = 0)

        last_X = X
        
        x_r_list.append(x_ref_seq[0])
        y_r_list.append(y_ref_seq[0])
        z_r_list.append(z_ref_seq[0])
                    
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

        end_time = time.time()
        print(end_time - start_time)

        ctrlCnt += 1

        rate.sleep()

    cdpr.setMotorVelo(0, 0, 0)


    # save trajectory datas
    np.savetxt(X_data_save_path, X_data_list)
    np.savetxt(U_data_save_path, U_data_list)
    np.savetxt(dX_data_save_path, dX_data_list)
    print('datas saved.')
    
    # plot
    fig = plt.figure(1)
    x_plot = fig.add_subplot(3,1,1)
    y_plot = fig.add_subplot(3,1,2)
    z_plot = fig.add_subplot(3,1,3)
    plt.ion()

    
    x_plot.plot(x_r_list)
    x_plot.plot(x_list)
    y_plot.plot(y_r_list)
    y_plot.plot(y_list)
    z_plot.plot(z_r_list)
    z_plot.plot(z_list)

    x_plot.set_ylabel('x')
    y_plot.set_ylabel('y')
    z_plot.set_ylabel('z')

    fig2 = plt.figure(2)
    traject_plot = fig2.add_subplot(1,1,1)
    traject_plot.plot(x_r_list, y_r_list)
    traject_plot.plot(x_list, y_list)

    
    plt.ioff()
    plt.show()
    

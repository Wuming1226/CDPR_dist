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



X_data_save_path = 'origin_data/2hz/08_X.txt'
U_data_save_path = 'origin_data/2hz/08_U.txt'
dX_data_save_path = 'origin_data/2hz/08_dX.txt'


'''
X: position
U: Motor Velocity
dX: position difference
'''


if __name__=="__main__":

    cdpr = CDPR()

    T = 0.5     # control period
    rate = rospy.Rate(1/T)
    
    x_r_list, y_r_list, z_r_list = [], [], []
    x_list, y_list, z_list = [], [], []

    X_data_list = np.zeros((0, 3))
    U_data_list = np.zeros((0, 3))
    dX_data_list = np.zeros((0, 3))

    x0, y0, z0, orient0 = cdpr.getMovingPlatformPose()
    last_X = np.array([x0, y0, z0])

    #############################################
    #               Model Training
    #############################################

    file_order = '01'
    X_path = 'origin_data/2hz/' + file_order + '_X.txt'
    U_path = 'origin_data/2hz/' + file_order + '_dX.txt'
    dX_path = 'origin_data/2hz/' + file_order + '_U.txt'

    X_data_list = np.loadtxt(X_path, dtype='float')
    U_data_list = np.loadtxt(U_path, dtype='float')
    dX_data_list = np.loadtxt(dX_path, dtype='float')
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
    plt.show()

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

    ##### control start #####

    ctrlCnt = 0
    refCnt = 0
    ratio = 1       # change referrence pose per 'ratio' control period

    cdpr.pretighten()

    while not rospy.is_shutdown() and ctrlCnt < 60:

        if not cdpr.check_in_workspace():
            cdpr.setMotorVelo(0, 0, 0)
            time.sleep(0.2)
            cdpr.setMotorVelo(0, 0, 0)
            exit()

        print(ctrlCnt)
        start_time = time.time()
        
        runTime = T * ctrlCnt
        
        # referrence pose of moment k  (unit: degree)
        t = refCnt * ratio * T
        x_r = 0.08 * math.sin(math.pi/15 * t) + cdpr.xOff
        y_r = 0.08 * math.cos(math.pi/15 * t) + cdpr.yOff
        z_r = -0.025 * math.sin(math.pi/7.5 * t) - 0.002 + cdpr.zOff
        pos_r = np.array([x_r, y_r, z_r])
        print('posRef: {}'.format(pos_r))

        ctrlCnt += 1
        if ctrlCnt % ratio == 0:
            refCnt += 1

        # referrence pose of moment k+1  (unit: degree)
        t = refCnt * ratio * T
        x_r_next = 0.08 * math.sin(math.pi/15 * t) + cdpr.xOff
        y_r_next = 0.08 * math.cos(math.pi/15 * t) + cdpr.yOff
        z_r_next = -0.025 * math.sin(math.pi/7.5 * t) - 0.002 + cdpr.zOff
        pos_r_next = np.array([x_r_next, y_r_next, z_r_next])

        # pose of moment k  (unit: degree)
        x, y, z, orientation = cdpr.getMovingPlatformPose()
        pos = np.array([x, y, z])
        print('pos: {}'.format(pos))

        # error of moment k  (unit: degree)
        pos_err = pos_r - pos

        # output velocity of moment k  (unit: degree/s)
        eps = 0.002
        k = 1.5
        veloTask = (pos_r_next - pos_r) / T + eps * np.sign(pos_err) + k * pos_err           # control law
        print('veloTask: {}'.format(veloTask))

        # inverse kinematics
        # calculate with model 
        # J = getJacobian(cdpr.getAnchorAPoses(), cdpr.getAnchorBPoses(), pos, orientation)
        # veloJoint = np.matmul(J, veloTask.reshape(3, 1))
        # veloJoint = veloJoint.reshape(3, )
        # print('veloJoint: {}'.format(veloJoint))
        # veloJoint = veloJoint*60*10/(0.03*math.pi)

        Xtensor = torch.from_numpy(X.reshape(1, 3)).to(device)
        MVtensor = torch.from_numpy((veloTask*0.5).reshape(1, 3)).to(device)
        dXtensor = model.predict_cuda(Xtensor, MVtensor)
        veloJoint = dXtensor.cpu().detach().numpy()
        #print(veloJoint)
        
        #############################################
        #                   sample
        #############################################

        # get the pose of the end effector
        x0, y0, z0, _ = cdpr.getMovingPlatformPose()
        X = np.array([x0, y0, z0])

        # set cable velocity in joint space
        for i in range(3):
            if np.abs(veloJoint[i]) > 500:
                veloJoint[i] = 500 * np.sign(veloJoint[i])
        set_start_time = time.time()
        cdpr.setMotorVelo(int(veloJoint[0]), int(veloJoint[1]), int(veloJoint[2]))
        print('motor velo: {}, {}, {}'.format(veloJoint[0], veloJoint[1], veloJoint[2]))
        print(time.time() - set_start_time)
        
        # generate input and output data (except for the first run)
        if ctrlCnt > 0:
            X_data = last_X
            U_data = np.array([int(veloJoint[0]), int(veloJoint[1]), int(veloJoint[2])])
            dX_data = X - last_X

            X_data_list = np.append(X_data_list, X_data.reshape(1, 3), axis = 0)
            U_data_list = np.append(U_data_list, U_data.reshape(1, 3), axis = 0)
            dX_data_list = np.append(dX_data_list, dX_data.reshape(1, 3), axis = 0)

        last_X = X
        
        x_r_list.append(x_r)
        y_r_list.append(y_r)
        z_r_list.append(z_r)
                    
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

        end_time = time.time()
        print(end_time - start_time)

        rate.sleep()

    cdpr.setMotorVelo(0, 0, 0)
    time.sleep(0.2)
    cdpr.setMotorVelo(0, 0, 0)
    time.sleep(0.2)
    cdpr.setMotorVelo(0, 0, 0)


    # save trajectory datas
    # np.savetxt(X_data_save_path, X_data_list)
    # np.savetxt(U_data_save_path, U_data_list)
    # np.savetxt(dX_data_save_path, dX_data_list)
    print('datas saved.')
    
    # plot
    fig = plt.figure(1)
    x_plot = fig.add_subplot(3,1,1)
    y_plot = fig.add_subplot(3,1,2)
    z_plot = fig.add_subplot(3,1,3)
    plt.ion()

    
    x_plot.plot(np.arange(0, ctrlCnt*T, T), x_r_list)
    x_plot.plot(np.arange(0, ctrlCnt*T, T), x_list)
    y_plot.plot(np.arange(0, ctrlCnt*T, T), y_r_list)
    y_plot.plot(np.arange(0, ctrlCnt*T, T), y_list)
    z_plot.plot(np.arange(0, ctrlCnt*T, T), z_r_list)
    z_plot.plot(np.arange(0, ctrlCnt*T, T), z_list)

    x_plot.set_ylabel('x')
    y_plot.set_ylabel('y')
    z_plot.set_ylabel('z')

    
    plt.ioff()
    plt.show()
    

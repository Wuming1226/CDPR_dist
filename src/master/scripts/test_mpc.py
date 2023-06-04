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

X_data_save_path = 'sim_traject_data/2k10010_X.txt'
U_data_save_path = 'sim_traject_data/01_U.txt'
dX_data_save_path = 'sim_traject_data/01_dX.txt'


'''
X: position
U: Motor Velocity
dX: position difference
'''

# anchors on the fixed frame (world frame)
anchorA1Pos = [0.718, 0.776, -0.073]
anchorA2Pos = [0.719, 0.077, -0.068]
anchorA3Pos = [0.061, 0.426, -0.056]
anchorAPos = [anchorA1Pos, anchorA2Pos, anchorA3Pos]

# anchors on the moving platform (body frame)
anchorB1Pos = [0, 0, 0]
anchorB2Pos = [0, 0, 0]
anchorB3Pos = [0, 0, 0]
anchorBPos = [anchorB1Pos, anchorB2Pos, anchorB3Pos]

# origin offset
xOff = 0.468
yOff = 0.418
zOff = -0.311


if __name__=="__main__":

    # cdpr = CDPR()
    rospy.init_node('cdpr_control', anonymous=True)

    T = 0.1     # control period
    rate = rospy.Rate(1/T)
    
    x_r_list, y_r_list, z_r_list = [], [], []
    x_list, y_list, z_list = [], [], []

    X_data_list = np.empty((0, 3))
    U_data_list = np.empty((0, 3))
    dX_data_list = np.empty((0, 3))

    last_X = np.array([0, 0, 0])


    #############################################
    #               Model Training
    #############################################

    file_order = '01'
    X_path = 'origin_data/random_collision/' + file_order + '_X.txt'
    U_path = 'origin_data/random_collision/' + file_order + '_U.txt'
    dX_path = 'origin_data/random_collision/' + file_order + '_dX.txt'

    X_data_list = np.loadtxt(X_path, dtype='float')
    U_data_list = np.loadtxt(U_path, dtype='float')
    dX_data_list = np.loadtxt(dX_path, dtype='float') * T

    for i in range(2, 10):
        file_order = '0{}'.format(i)
        X_path = 'origin_data/random_collision/' + file_order + '_X.txt'
        U_path = 'origin_data/random_collision/' + file_order + '_U.txt'
        dX_path = 'origin_data/random_collision/' + file_order + '_dX.txt'

        X_new = np.loadtxt(X_path, dtype='float')
        U_new = np.loadtxt(U_path, dtype='float')
        dX_new = np.loadtxt(dX_path, dtype='float') * T

        X_data_list = np.append(X_data_list, X_new, axis = 0)
        U_data_list = np.append(U_data_list, U_new, axis = 0)
        dX_data_list = np.append(dX_data_list, dX_new, axis = 0)

    for i in range(11, 14):
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

    real_x = xOff
    real_y = 0.1 + yOff
    real_z = zOff

    # cdpr.pretighten()

    while not rospy.is_shutdown() and ctrlCnt < 40 / T:

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
            # x_ref_seq = np.append(x_ref_seq, 0.12 * math.sin(math.pi/10 * (t + i)))
            # y_ref_seq = np.append(y_ref_seq, 0.12 * math.cos(math.pi/10 * (t + i)))
            # z_ref_seq = np.append(z_ref_seq, 0)
            x_ref_seq = np.append(x_ref_seq, 0.10 * math.sin(math.pi/20 * (t + i * T)) + xOff)
            y_ref_seq = np.append(y_ref_seq, 0.10 * math.cos(math.pi/20 * (t + i * T)) + yOff)
            z_ref_seq = np.append(z_ref_seq, 0.08 * math.sin(math.pi/10 * (t + i * T)) + zOff)
        print('target(position): {}'.format([x_ref_seq[0], y_ref_seq[0], z_ref_seq[0]]))


        # pose of moment k  (unit: degree)
        # x, y, z, orientation = cdpr.getMovingPlatformPose()
        x, y, z, orientation =[real_x, real_y, real_z, [0, 0, 0, 1]]
        X = np.array([x, y, z])
        print('state(position): {}'.format(X))


        #############################################
        #               MPC Controller
        #############################################
        
        mpc_time = time.time()
        action = mpc.get_action(x_ref_seq, y_ref_seq, z_ref_seq, X)
        print('mpc time: {}'.format(time.time() - mpc_time))
        motor_velo = action
        print("action(motor velo): {}".format(motor_velo))
        

        #############################################
        #                   sample
        #############################################

        # get the pose of the end effector
        x0, y0, z0,_ =[real_x, real_y, real_z, [0, 0, 0, 1]]

        # x0, y0, z0, _ = cdpr.getMovingPlatformPose()
        X = np.array([x0, y0, z0])

        # real_x = real_x + 0.001 * action[0] - 0.001 * action[1]
        # real_y = real_y + 0.001 * action[2] - 0.001 * action[3]
        # Xtensor = torch.from_numpy(X.reshape(1, 3)).to(device)
        # MVtensor = torch.from_numpy(motor_velo.reshape(1, 3)).to(device)
        # predict_time = time.time()
        # dXtensor = model.predict_cuda(Xtensor, MVtensor)
        # print('predict time: {}'.format(time.time()-predict_time))
        # real_x += dXtensor.cpu().detach().numpy()[0]
        # real_y += dXtensor.cpu().detach().numpy()[1]
        # real_z += dXtensor.cpu().detach().numpy()[2]

        J = getJacobian(anchorAPos, anchorBPos, X, np.array([0, 0, 0, 1]))
        dx = np.matmul(np.linalg.inv(J), motor_velo.reshape(3, 1)/600*(0.03*np.pi))
        dx = dx.reshape(1, 3) * T
        print(dx)
        real_x += dx[0, 0]
        real_y += dx[0, 1]
        real_z += dx[0, 2]


        # generate input and output data (except for the first run)
        if ctrlCnt > 0:
            X_data = last_X
            U_data = last_U
            dX_data = X - last_X

            X_data_list = np.append(X_data_list, X_data.reshape(1, 3), axis = 0)
            U_data_list = np.append(U_data_list, U_data.reshape(1, 3), axis = 0)
            dX_data_list = np.append(dX_data_list, dX_data.reshape(1, 3), axis = 0)

        last_X = X
        last_U = np.array([int(motor_velo[0]), int(motor_velo[1]), int(motor_velo[2])])
        
        x_r_list.append(x_ref_seq[0])
        y_r_list.append(y_ref_seq[0])
        z_r_list.append(z_ref_seq[0])
                    
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

        end_time = time.time()
        print('run time: {}'.format(end_time - start_time))

        ctrlCnt += 1

        rate.sleep()

    # cdpr.setMotorVelo(0, 0, 0)


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

    fig2 = plt.figure(2)
    traject_plot = fig2.add_subplot(1,1,1)
    traject_plot.plot(x_r_list, y_r_list)
    traject_plot.plot(x_list, y_list)

    
    plt.ioff()
    plt.show()
    

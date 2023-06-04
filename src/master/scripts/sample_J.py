#! /usr/bin/env python3

import time
import numpy as np
import math
import rospy
import matplotlib.pyplot as plt
import scipy.io as io

from cdpr import CDPR
from transform import euler2quaternion
from Jacobian import getJacobian

X_data_save_path = 'origin_data/10hz/02_X.txt'
U_data_save_path = 'origin_data/10hz/02_U.txt'
dX_data_save_path = 'origin_data/10hz/02_dX.txt'
# X_data_save_path = 'collision_origin_data/10hz/01_X.txt'
# U_data_save_path = 'collision_origin_data/10hz/01_U.txt'
# dX_data_save_path = 'collision_origin_data/10hz/01_dX.txt'


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

    X_data_list = np.zeros((0, 3))
    U_data_list = np.zeros((0, 3))
    dX_data_list = np.zeros((0, 3))

    x0, y0, z0, orient0 = cdpr.getMovingPlatformPose()
    last_X = np.array([x0, y0, z0])

    ##### control start #####

    cnt = 0

    cdpr.pretighten()

    while not rospy.is_shutdown() and cnt < 40 / T:

        if not cdpr.check_in_workspace():
            cdpr.setMotorVelo(0, 0, 0)
            time.sleep(0.2)
            cdpr.setMotorVelo(0, 0, 0)
            exit()

        print(cnt)
        start_time = time.time()
        
        runTime = T * cnt
        
        # referrence pose of moment k  (unit: degree)
        t = cnt * T
        x_r = 0.1 * math.sin(math.pi/20 * t) + cdpr.xOff
        y_r = 0.1 * math.cos(math.pi/20 * t) + cdpr.yOff
        z_r = 0.08 * math.sin(math.pi/10 * t) + 0 + cdpr.zOff
        # side = 0.3
        # wave = 0.0
        # if cnt < 100:
        #     x_r = -side / 2 + side / 100 * (cnt % 100) + cdpr.xOff
        #     y_r = 0.2 + cdpr.yOff
        #     z_r = -wave / 2 + wave / 100 * (cnt % 100) + cdpr.zOff
        # elif cnt < 200:
        #     x_r = side / 2 + cdpr.xOff
        #     y_r = side / 2 - side / 100 * (cnt % 100) + cdpr.yOff
        #     z_r = wave / 2 - wave / 100 * (cnt % 100) + cdpr.zOff
        # elif cnt < 300:
        #     x_r = side / 2 - side / 100 * (cnt % 100) + cdpr.xOff
        #     y_r = -side / 2 + cdpr.yOff
        #     z_r = -wave / 2 + wave / 100 * (cnt % 100) + cdpr.zOff
        # elif cnt < 400:
        #     x_r = -side / 2 + cdpr.xOff
        #     y_r = -side / 2 + side / 100 * (cnt % 100) + cdpr.yOff
        #     z_r = wave / 2 - wave / 100 * (cnt % 100) + cdpr.zOff
        pos_r = np.array([x_r, y_r, z_r])
        print('posRef: {}'.format(pos_r))

        # referrence pose of moment k+1  (unit: degree)
        t = cnt * T
        x_r_next = 0.1 * math.sin(math.pi/20 * t) + cdpr.xOff
        y_r_next = 0.1 * math.cos(math.pi/20 * t) + cdpr.yOff
        z_r_next = 0.08 * math.sin(math.pi/10 * t) + 0 + cdpr.zOff
        # if cnt < 100:
        #     x_r_next = -side / 2 + side / 100 * (cnt % 100) + cdpr.xOff
        #     y_r_next = 0.2 + cdpr.yOff
        #     z_r_next = -wave / 2 + wave / 100 * (cnt % 100) + cdpr.zOff
        # elif cnt < 200:
        #     x_r_next = side / 2 + cdpr.xOff
        #     y_r_next = side / 2 - side / 100 * (cnt % 100) + cdpr.yOff
        #     z_r_next = wave / 2 - wave / 100 * (cnt % 100) + cdpr.zOff
        # elif cnt < 300:
        #     x_r_next = side / 2 - side / 100 * (cnt % 100) + cdpr.xOff
        #     y_r_next = -side / 2 + cdpr.yOff
        #     z_r_next = -wave / 2 + wave / 100 * (cnt % 100) + cdpr.zOff
        # elif cnt < 400:
        #     x_r_next = -side / 2 + cdpr.xOff
        #     y_r_next = -side / 2 + side / 100 * (cnt % 100) + cdpr.yOff
        #     z_r_next = wave / 2 - wave / 100 * (cnt % 100) + cdpr.zOff
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
        J = getJacobian(cdpr.getAnchorAPoses(), cdpr.getAnchorBPoses(), pos, orientation)
        veloJoint = np.matmul(J, veloTask.reshape(3, 1))
        veloJoint = veloJoint.reshape(3, )
        print('veloJoint: {}'.format(veloJoint))

        veloJoint = veloJoint*60*10/(0.03*math.pi)
        #print(veloJoint)
        
        #############################################
        #                   sample
        #############################################

        # get the pose of the end effector
        x0, y0, z0, _ = cdpr.getMovingPlatformPose()
        X = np.array([x0, y0, z0])

        # set cable velocity in joint space
        for i in range(2):
            if np.abs(veloJoint[i]) > 250:
                veloJoint[i] = 250 * np.sign(veloJoint[i])
        if np.abs(veloJoint[2]) > 350:
                veloJoint[2] = 350 * np.sign(veloJoint[i])
        set_start_time = time.time()
        cdpr.setMotorVelo(int(veloJoint[0]), int(veloJoint[1]), int(veloJoint[2]))
        print('motor velo: {}, {}, {}'.format(veloJoint[0], veloJoint[1], veloJoint[2]))
        print(time.time() - set_start_time)
        
        # generate input and output data (except for the first run)
        if cnt > 0:
            X_data = last_X
            U_data = last_U
            dX_data = (X - last_X) / T

            X_data_list = np.append(X_data_list, X_data.reshape(1, 3), axis = 0)
            U_data_list = np.append(U_data_list, U_data.reshape(1, 3), axis = 0)
            dX_data_list = np.append(dX_data_list, dX_data.reshape(1, 3), axis = 0)

        last_X = X
        last_U = np.array([int(veloJoint[0]), int(veloJoint[1]), int(veloJoint[2])])
        
        x_r_list.append(x_r)
        y_r_list.append(y_r)
        z_r_list.append(z_r)
                    
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

        end_time = time.time()
        print(end_time - start_time)

        cnt += 1
        rate.sleep()

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

    
    x_plot.plot(np.arange(0, cnt*T, T), x_r_list)
    x_plot.plot(np.arange(0, cnt*T, T), x_list)
    y_plot.plot(np.arange(0, cnt*T, T), y_r_list)
    y_plot.plot(np.arange(0, cnt*T, T), y_list)
    z_plot.plot(np.arange(0, cnt*T, T), z_r_list)
    z_plot.plot(np.arange(0, cnt*T, T), z_list)

    x_plot.set_ylabel('x')
    y_plot.set_ylabel('y')
    z_plot.set_ylabel('z')

    
    plt.ioff()
    plt.show()
    

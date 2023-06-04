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

X_data_save_path = 'origin_data/random_collision/12_X.txt'
U_data_save_path = 'origin_data/random_collision/12_U.txt'
dX_data_save_path = 'origin_data/random_collision/12_dX.txt'
# X_data_save_path = 'collision_origin_data/10hz/01_X.txt'
# U_data_save_path = 'collision_origin_data/10hz/01_U.txt'
# dX_data_save_path = 'collision_origin_data/10hz/01_dX.txt'


'''
X: position
U: Motor Velocity
dX: position difference
'''

def cdpr_go_to(target):
    global cdpr
    global rate

    print('---------------------------------------')
    print('going to: {}'.format(target))

    while not rospy.is_shutdown():

        if not cdpr.check_in_workspace():
            cdpr.setMotorVelo(0, 0, 0)
            print('TRESPASS')
            exit()
        
        # referrence pose of moment k  (unit: degree)
        x_r = target[0]
        y_r = target[1]
        z_r = target[2]
        pos_r = np.array([x_r, y_r, z_r])

        # referrence pose of moment k+1  (unit: degree)
        x_r_next = target[0]
        y_r_next = target[1]
        z_r_next = target[2]
        pos_r_next = np.array([x_r_next, y_r_next, z_r_next])

        # pose of moment k  (unit: degree)
        x, y, z, orientation = cdpr.getMovingPlatformPose()
        pos = np.array([x, y, z])
        print('pos: {}'.format(pos))
        if pos_r[0] - pos[0] < 0.01 and pos_r[1] - pos[1] < 0.01 and pos_r[2] - pos[2] < 0.01:
            cdpr.setMotorVelo(0, 0, 0)
            break

        # error of moment k  (unit: degree)
        pos_err = pos_r - pos

        # output velocity of moment k  (unit: degree/s)
        eps = 0.002
        k = 1.5
        veloTask = (pos_r_next - pos_r) / T + eps * np.sign(pos_err) + k * pos_err           # control law

        # inverse kinematics
        J = getJacobian(cdpr.getAnchorAPoses(), cdpr.getAnchorBPoses(), pos, orientation)
        veloJoint = np.matmul(J, veloTask.reshape(3, 1))
        veloJoint = veloJoint.reshape(3, )

        veloJoint = veloJoint*60*10/(0.03*math.pi)

        # set cable velocity in joint space
        for i in range(3):
            if np.abs(veloJoint[i]) > 500:
                veloJoint[i] = 500 * np.sign(veloJoint[i])
        cdpr.setMotorVelo(int(veloJoint[0]), int(veloJoint[1]), int(veloJoint[2]))

        rate.sleep()


def check_in_workspace(x, y, z):
    global cdpr

    if z > cdpr.zOff + 0.12 or z < cdpr.zOff - 0.12:
        return False
    else:
        if x > cdpr._anchorA1Pos[0] or x < cdpr._anchorA3Pos[0]:
            return False
        else:
            if (y - cdpr._anchorA3Pos[1]) / (x - cdpr._anchorA3Pos[0]) > (cdpr._anchorA1Pos[1] - 0.10 - cdpr._anchorA3Pos[1]) / (cdpr._anchorA1Pos[0] - cdpr._anchorA3Pos[0]) \
            or (y - cdpr._anchorA3Pos[1]) / (x - cdpr._anchorA3Pos[0]) < (cdpr._anchorA2Pos[1] + 0.10 - cdpr._anchorA3Pos[1]) / (cdpr._anchorA2Pos[0] - cdpr._anchorA3Pos[0]):
                return False
            else:
                if (x - cdpr.xOff) ** 2 + (y - cdpr.yOff) ** 2 > 0.16 ** 2:
                    return False
                else:
                    return True


if __name__=="__main__":

    cdpr = CDPR()

    T = 0.5     # control period
    rate = rospy.Rate(1/T)

    X_data_list = np.zeros((0, 3))
    U_data_list = np.zeros((0, 3))
    dX_data_list = np.zeros((0, 3))

    x0, y0, z0, orient0 = cdpr.getMovingPlatformPose()
    last_X = np.array([x0, y0, z0])

    ##### control start #####

    cnt = 0

    cdpr.pretighten()

    while not rospy.is_shutdown() and cnt < 100:

        print(cnt)
        start_time = time.time()

        # generate position
        x, y, z = 0, 0, 0
        while not check_in_workspace(x, y, z):
            x = np.random.uniform(low=cdpr._anchorA3Pos[0]+0.15, high=cdpr._anchorA1Pos[0]-0.15)
            y = np.random.uniform(low=cdpr._anchorA2Pos[1]+0.05, high=cdpr._anchorA1Pos[1]-0.05)
            z = np.random.uniform(low=-0.15+cdpr.zOff, high=0.15+cdpr.zOff)

        cdpr_go_to(np.array([x, y, z]))
        time.sleep(1)

        # generate velocity
        v1 = np.random.uniform(low=-250, high=250)
        v2 = np.random.uniform(low=-250, high=250)
        v3 = np.random.uniform(low=-350, high=350)
        veloTask = np.array([v1, v2, v3])
    
        
        #############################################
        #                   sample
        #############################################

        # get the pose of the end effector
        x0, y0, z0, _ = cdpr.getMovingPlatformPose()
        X0 = np.array([x0, y0, z0])

        # set cable velocity in joint space
        cdpr.setMotorVelo(int(v1), int(v2), int(v3))
        print('motor velo: {}, {}, {}'.format(v1, v2, v3))
        start_time = time.time()

        time.sleep(0.1)
        cdpr.setMotorVelo(0, 0, 0)
        time.sleep(0.05)

        end_time = time.time()
        x, y, z, _ = cdpr.getMovingPlatformPose()
        X = np.array([x, y, z])

        
        # generate input and output data (except for the first run)
        X_data = X0
        U_data = np.array([int(v1), int(v2), int(v3)])
        dX_data = (X - X0) / (end_time - start_time)

        X_data_list = np.append(X_data_list, X_data.reshape(1, 3), axis = 0)
        U_data_list = np.append(U_data_list, U_data.reshape(1, 3), axis = 0)
        dX_data_list = np.append(dX_data_list, dX_data.reshape(1, 3), axis = 0)

        # save data
        np.savetxt(X_data_save_path, X_data_list)
        np.savetxt(U_data_save_path, U_data_list)
        np.savetxt(dX_data_save_path, dX_data_list)
        print('data saved.')

        end_time = time.time()
        print(end_time - start_time)
        time.sleep(1)

        cnt += 1

    cdpr.setMotorVelo(0, 0, 0)


    
    # # plot
    # fig = plt.figure(1)
    # x_plot = fig.add_subplot(3,1,1)
    # y_plot = fig.add_subplot(3,1,2)
    # z_plot = fig.add_subplot(3,1,3)
    # plt.ion()

    
    # x_plot.plot(np.arange(0, cnt*T, T), x_r_list)
    # x_plot.plot(np.arange(0, cnt*T, T), x_list)
    # y_plot.plot(np.arange(0, cnt*T, T), y_r_list)
    # y_plot.plot(np.arange(0, cnt*T, T), y_list)
    # z_plot.plot(np.arange(0, cnt*T, T), z_r_list)
    # z_plot.plot(np.arange(0, cnt*T, T), z_list)

    # x_plot.set_ylabel('x')
    # y_plot.set_ylabel('y')
    # z_plot.set_ylabel('z')

    
    plt.ioff()
    plt.show()
    

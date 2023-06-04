#! /usr/bin/env python3

import time
import numpy as np
import math
import rospy
import matplotlib.pyplot as plt
import scipy.io as io

from transform import euler2quaternion
from Jacobian import getJacobian


X_data_save_path = 'sim_origin_data/5hz/200_X.txt'
U_data_save_path = 'sim_origin_data/5hz/200_U.txt'
dX_data_save_path = 'sim_origin_data/5hz/200_dX.txt'


'''
X: position
U: Motor Velocity
dX: position difference
'''


# anchors on the fixed frame (world frame)
anchorA1Pos = [0.717, 0.770, -0.052]
anchorA2Pos = [0.699, 0.058, -0.050]
anchorA3Pos = [0.064, 0.425, -0.040]
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

def check_in_workspace(x, y, z):
    if z > zOff + 0.1 or z < zOff - 0.1:
        return False
    else:
        if x > anchorA1Pos[0] or x < anchorA3Pos[0]:
            return False
        else:
            if (y - anchorA3Pos[1]) / (x - anchorA3Pos[0]) > (anchorA1Pos[1] - anchorA3Pos[1]) / (anchorA1Pos[0] - anchorA3Pos[0]) \
            or (y - anchorA3Pos[1]) / (x - anchorA3Pos[0]) < (anchorA2Pos[1] - anchorA3Pos[1]) / (anchorA2Pos[0] - anchorA3Pos[0]):
                return False
            else:
                return True
            

if __name__=="__main__":

    X_data_list = np.zeros((0, 3))
    U_data_list = np.zeros((0, 3))
    dX_data_list = np.zeros((0, 3))

    cnt = 0

    while not rospy.is_shutdown() and cnt < 200:

        print('-----------------------------------------')
        print('              sample: {}                 '.format(cnt))
        # generate position
        x, y, z = 0, 0, 0
        while not check_in_workspace(x, y, z):
            x = np.random.uniform(low=anchorA3Pos[0], high=anchorA1Pos[0])
            y = np.random.uniform(low=anchorA2Pos[1], high=anchorA1Pos[1])
            z = np.random.uniform(low=-0.1+zOff, high=0.1+zOff)

        pos = np.array([x, y, z])
        print('pos: {}'.format(pos))

        # generate velocity
        vx = np.random.uniform(low=-0.04, high=0.04)
        vy = np.random.uniform(low=-0.04, high=0.04)
        vz = np.random.uniform(low=-0.04, high=0.04)
        veloTask = np.array([vx, vy, vz])
        print('veloTask: {}'.format(veloTask))

        # inverse kinematics
        # calculate with model 
        J = getJacobian(anchorAPos, anchorBPos, pos, np.array([0, 0, 0, 1]))
        veloJoint = np.matmul(J, veloTask.reshape(3, 1))

        veloJoint = veloJoint*60*10/(0.03*math.pi)
        print('veloJoint: {}'.format(veloJoint))

        # generate input and output data (except for the first run)
        X_data = pos
        U_data = veloJoint
        dX_data = veloTask

        X_data_list = np.append(X_data_list, X_data.reshape(1, 3), axis = 0)
        U_data_list = np.append(U_data_list, U_data.reshape(1, 3), axis = 0)
        dX_data_list = np.append(dX_data_list, dX_data.reshape(1, 3), axis = 0)

        cnt += 1
        

    # save trajectory datas
    np.savetxt(X_data_save_path, X_data_list)
    np.savetxt(U_data_save_path, U_data_list)
    np.savetxt(dX_data_save_path, dX_data_list)
    print('datas saved.')
    

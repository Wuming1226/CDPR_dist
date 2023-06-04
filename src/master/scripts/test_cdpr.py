#! /usr/bin/env python3

import time
import numpy as np
import math
import rospy
import scipy.io as io
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cdpr import CDPR
from transform import euler2quaternion
from Jacobian import getJacobian



if __name__=="__main__":

    cdpr = CDPR()

    T = 0.4     # control period
    rate = rospy.Rate(1/T)
    
    xRefList, yRefList, zRefList = [], [], []
    xList, yList, zList = [], [], []
    xErrList, yErrList, zErrList = [], [], []

    traject_data = np.zeros((1,7))
    
    Setpoint = []


    ##### control start #####

    ctrlCnt = 0
    refCnt = 0
    ratio = 1       # change referrence pose per 'ratio' control period


    while not rospy.is_shutdown() and ctrlCnt < 200:

        print(ctrlCnt)
        start_time = time.time()
        
        runTime = T * ctrlCnt
        
        # referrence pose of moment k  (unit: degree)
        t = refCnt * ratio * T
        xRef = 0.5 * math.sin(2*math.pi/40 * T * ctrlCnt)
        yRef = 0.5 * math.cos(2*math.pi/40 * T * ctrlCnt)
        zRef = 0.1 * math.sin(2*math.pi/20 * T * ctrlCnt)
        pos = np.array([x, y, z])
        print(xRef, yRef, zRef)
        posRef = np.array([xRef, yRef, zRef])

        ctrlCnt += 1
        if ctrlCnt % ratio == 0:
            refCnt += 1

        # referrence pose of moment k+1  (unit: m)
        t = refCnt * ratio * T
        xRefNext = 0.5 * math.sin(2*math.pi/40 * T * ctrlCnt)
        yRefNext = 0.5 * math.cos(2*math.pi/40 * T * ctrlCnt)
        zRefNext = 0.1 * math.sin(2*math.pi/20 * T * ctrlCnt)
        posRefNext = np.array([xRefNext, yRefNext, zRefNext])

        # moving platform pose of moment k  (unit: m)
        x, y, z, orientation = cdpr.getMovingPlatformPose()
        pos = np.array([x, y, z])

        # error of moving platform pose of moment k  (unit: m)
        posErr = posRef - pos

        # output velocity in task space of moment k  (unit: m/s)
        eps = 0.01         # dmax
        k1 = 2.5
        k2 = 2.5
        k3 = 2.5
        veloTask = (posRefNext - posRef) / T + eps * np.sign(posErr) + np.array([k1, k2, k3]) * posErr       # control law

        # inverse kinematics
        J = getJacobian(cdpr.getAnchorAPoses(), cdpr.getAnchorBPoses(), pos, orientation)
        veloJoint = np.matmul(J, veloTask.reshape(3, 1))
        print(veloJoint)


        # set actuator velocity
        cdpr.setMotorVelo(int(veloJoint[0]*60/(0.03*math.pi)*10), int(veloJoint[1]*60/(0.03*math.pi)*10), int(veloJoint[2]*60/(0.03*math.pi)*10), int(veloJoint[3]*60/(0.03*math.pi)*10))
        print(int(veloJoint[0]*60/(0.03*math.pi)*10), int(veloJoint[1]*60/(0.03*math.pi)*10), int(veloJoint[2]*60/(0.03*math.pi)*10), int(veloJoint[3]*60/(0.03*math.pi)*10))

        rate.sleep()
            

    cdpr.setMotorVelo(0, 0)

    
    
    

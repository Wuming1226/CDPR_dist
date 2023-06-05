#! /usr/bin/env python3

import os
import rospy
import time

from motor import Motor

from std_msgs.msg import Int16MultiArray
from slave.srv import SetMotorVelo, SetMotorVeloResponse, GetMotorPos, GetMotorPosResponse


class CDPR:
    def __init__(self, motor1ID, motor2ID, motor3ID, motor4ID):

        # can settings
        os.system('sudo ip link set can0 type can bitrate 500000')
        os.system('sudo ifconfig can0 up')
        self.canChannel = 'can0'

        # initialize motors
        self._motor1 = Motor(self.canChannel, motor1ID)
        self._motor1.setProfVeloMode(0)
        self._motor2 = Motor(self.canChannel, motor2ID)
        self._motor2.setProfVeloMode(0)
        self._motor3 = Motor(self.canChannel, motor3ID)
        self._motor3.setProfVeloMode(0)
        self._motor4 = Motor(self.canChannel, motor4ID)
        self._motor4.setProfVeloMode(0)


        # ros settings
        rospy.init_node('cdpr_actuate', anonymous=True)
        self._setMotorVeloSrv = rospy.Service('set_motor_velo', SetMotorVelo, self._setMotorVeloHandle)
        self._getMotorPosSrv = rospy.Service('get_motor_pos', GetMotorPos, self._getMotorPosHandle)
        rospy.Subscriber('motor_velo', Int16MultiArray, callback=self.velo_callback)
        
        
    def velo_callback(self, msg):
        self._motor1.setVelo(msg.data[0])
        self._motor2.setVelo(msg.data[1])
        self._motor3.setVelo(msg.data[2])
        print(msg.data[0], msg.data[1], msg.data[2])
    # def receiveMotorVelo(self):
    #     msg = rospy.wait_for_message('motor_velo', Int16MultiArray, timeout=None)
    #     self._motor1.setVelo(msg.data[0])
    #     self._motor2.setVelo(msg.data[1])
    #     self._motor3.setVelo(msg.data[2])
    #     print(msg.data[0], msg.data[1], msg.data[2])
    # def _setMotorVeloHandle(self, req):
    #     self._motor1.setVelo(req.motor1Velo)
    #     self._motor2.setVelo(req.motor2Velo)
    #     self._motor3.setVelo(req.motor3Velo)
    #     print(req.motor1Velo, req.motor2Velo, req.motor3Velo)
    #     return SetMotorVeloResponse(True)


    ########## to be updated ##########
    def _getMotorPosHandle(self, req):
        print(self._motor1.getPos(), self._motor2.getPos(), self._motor3.getPos(), self._motor4.getPos())
        return GetMotorPosResponse(self._motor1.getPos(), self._motor2.getPos(), self._motor3.getPos(), self._motor4.getPos())
    ########## to be updated ##########



if __name__=="__main__":
    cdpr = CDPR(motor1ID=2, motor2ID=4, motor3ID=3)

    rate = rospy.Rate(10)
    cnt = 0
    while not rospy.is_shutdown():
        #cdpr.receiveMotorVelo()
        rate.sleep()

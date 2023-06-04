#! /usr/bin/env python3

import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int16MultiArray


def listener2():
    rospy.init_node('listener', anonymous=True)
    rate = rospy.Rate(10)
    #用循环来订阅所有数据
    while not rospy.is_shutdown():
        #订阅话题
        msg = rospy.wait_for_message('motor_velo', Int16MultiArray, timeout=None)
        print(msg.data)
        rate.sleep()

if __name__ == '__main__':

    #运行程序2
    
    listener2()

#! /usr/bin/env python3

import rospy
import time
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int16MultiArray

from slave.srv import SetMotorVelo, GetMotorPos

from transform import quaternion2euler


class CDPR:
    def __init__(self):
        # ros settings
        rospy.init_node('cdpr_control', anonymous=True)

        # origin offset
        # self.xOff = 0.473
        # self.yOff = 0.420
        # self.zOff = -0.337 # - 0.3
        self.xOff = 0.473 
        self.yOff = 0.420
        self.zOff = -0.490 - 0.025
        
        # pose and limit of the end effector
        self._movingPlatformPose = PoseStamped()
        rospy.Subscriber('/vrpn_client_node/platform/pose', PoseStamped, self._poseCallback, queue_size=1)
        self.veloPub = rospy.Publisher('motor_velo', Int16MultiArray, queue_size=10)

        # anchors on the fixed frame (world frame)
        self._anchorA1Pos = [0.717, 0.770, -0.052]
        self._anchorA2Pos = [0.699, 0.058, -0.050]
        self._anchorA3Pos = [0.064, 0.425, -0.040]

        # anchors on the moving platform (body frame)
        self._anchorB1Pos = [0, 0, 0]
        self._anchorB2Pos = [0, 0, 0]
        self._anchorB3Pos = [0, 0, 0]


    def _poseCallback(self, data):
        # if motion data is lost(999999), do not update
        if np.abs(data.pose.position.x) > 2000 or np.abs(data.pose.position.y) > 2000 or np.abs(data.pose.position.z) > 2000:
            pass
        else:
            # pose
            self._movingPlatformPose.pose.position.x = data.pose.position.x / 1000
            self._movingPlatformPose.pose.position.y = data.pose.position.y / 1000
            self._movingPlatformPose.pose.position.z = data.pose.position.z / 1000
            self._movingPlatformPose.pose.orientation = data.pose.orientation

            # header
            self._movingPlatformPose.header.frame_id = data.header.frame_id
            self._movingPlatformPose.header.stamp = data.header.stamp


    def setMotorVelo(self, motor1Velo, motor2Velo, motor3Velo):
        # rospy.wait_for_service('set_motor_velo')
        # try:
        #     set_motor_velo = rospy.ServiceProxy('set_motor_velo', SetMotorVelo)
        #     set_motor_velo(motor1Velo, motor2Velo, motor3Velo)
        # except rospy.ServiceException as e:
        #     print("Service call failed: {}".format(e))
        motor_velo = Int16MultiArray(data=np.array([motor1Velo, motor2Velo, motor3Velo]))
        self.veloPub.publish(motor_velo)

    
    def getMotorPos(self):
        rospy.wait_for_service('set_motor_velo')
        try:
            get_motor_pos = rospy.ServiceProxy('get_motor_pos', GetMotorPos)
            # print('1: {}'.format(time.time()))
            response = get_motor_pos()
            print(response.motor1Pos, response.motor2Pos, response.motor3Pos)
            return [response.motor1Pos, response.motor2Pos, response.motor3Pos]
            # print('2: {}'.format(time.time()))
        except rospy.ServiceException as e:
            print("Service call failed: {}".format(e))


    def getMovingPlatformPose(self):
        return self._movingPlatformPose.pose.position.x, self._movingPlatformPose.pose.position.y, self._movingPlatformPose.pose.position.z,\
            [self._movingPlatformPose.pose.orientation.x, self._movingPlatformPose.pose.orientation.y, self._movingPlatformPose.pose.orientation.z, self._movingPlatformPose.pose.orientation.w]

    def check_in_workspace(self):
        x, y, z, _ = self.getMovingPlatformPose()
        if z > self.zOff + 0.15 or z < self.zOff - 0.15:
            return False
        else:
            if x > self._anchorA1Pos[0] or x < self._anchorA3Pos[0]:
                return False
            else:
                if (y - self._anchorA3Pos[1]) / (x - self._anchorA3Pos[0]) > (self._anchorA1Pos[1] - self._anchorA3Pos[1]) / (self._anchorA1Pos[0] - self._anchorA3Pos[0]) \
                or (y - self._anchorA3Pos[1]) / (x - self._anchorA3Pos[0]) < (self._anchorA2Pos[1] - self._anchorA3Pos[1]) / (self._anchorA2Pos[0] - self._anchorA3Pos[0]):
                    return False
                else:
                    if (x - self.xOff) ** 2 + (y - self.yOff) ** 2 > 0.16 ** 2:
                        return False
                    else:
                        return True


    def pretighten(self):
        time.sleep(0.5)
        # cable1 pre-tightening
        print('cable1 pretightening...')
        self.setMotorVelo(-50, 0, 0)
        x0, y0, z0, _ = self.getMovingPlatformPose()
        while True:
            x, y, z, _ = self.getMovingPlatformPose()
            if np.linalg.norm(np.array([x,y,z]) - np.array([x0,y0,z0]), ord=2) > 0.003:
                self.setMotorVelo(0, 0, 0)
                break
            else:
                time.sleep(0.1)
        #print('cable1 pretightened') 
        time.sleep(0.5)

        # cable2 pre-tightening
        print('cable2 pretightening...')
        self.setMotorVelo(0, -50, 0)
        x0, y0, z0, _ = self.getMovingPlatformPose()
        while True:
            x, y, z, _ = self.getMovingPlatformPose()
            if np.linalg.norm(np.array([x,y,z]) - np.array([x0,y0,z0]), ord=2) > 0.003:
                self.setMotorVelo(0, 0, 0)
                break
            else:
                time.sleep(0.1)
        #print('cable2 pretightened') 
        time.sleep(0.5)

        # cable3 pre-tightening
        print('cable3 pretightening...')
        self.setMotorVelo(0, 0, -50)
        x0, y0, z0, _ = self.getMovingPlatformPose()
        while True:
            x, y, z, _ = self.getMovingPlatformPose()
            if np.linalg.norm(np.array([x,y,z]) - np.array([x0,y0,z0]), ord=2) > 0.003:
                self.setMotorVelo(0, 0, 0)
                break
            else:
                time.sleep(0.1)
        #print('cable1 pretightened') 
        time.sleep(0.5)
        

    def loosen(self):
        print('loosening...')
        self.setMotorVelo(600, 600, 600)
        time.sleep(0.2)
        self.setMotorVelo(0, 0, 0)
        time.sleep(0.5)
        

    def getAnchorAPoses(self):
        return [self._anchorA1Pos, self._anchorA2Pos, self._anchorA3Pos]


    def getAnchorBPoses(self):
        return [self._anchorB1Pos, self._anchorB2Pos, self._anchorB3Pos]



if __name__=="__main__":

    cdpr = CDPR()
    rate = rospy.Rate(10)
    cdpr.pretighten()
    # while True:
    #     print(cdpr.check_in_workspace())
    #     x, y, z, _  = cdpr.getMovingPlatformPose()
    #     print(x, y, z)
    #     cdpr.setMotorVelo(20, 10, 10)
    # cdpr.loosen()
    # time.sleep(1)
    # cdpr.setMotorVelo(-50, 0, 0)
    # time.sleep(1)
    # cdpr.setMotorVelo(0, 0, 0)
    # time.sleep(1)
    # cdpr.setMotorVelo(0, 0, 50)
    # time.sleep(1)
    # cdpr.setMotorVelo(0, 0, 0)
    # time.sleep(1)
    # cdpr.setMotorVelo(int(0.03 * 60/(0.03*np.pi)*10), int(0.01 * 60/(0.03*np.pi)*10), int(0.01 * 60/(0.03*np.pi)*10))
    # time.sleep(1)
    # cdpr.setMotorVelo(0, 0, 0)

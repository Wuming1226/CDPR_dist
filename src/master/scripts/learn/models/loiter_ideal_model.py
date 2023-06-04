#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# This script is trying to hover a quadrotor with a random MPC base on ideal model

import numpy as np
from datetime import datetime
import random
from scipy.spatial.transform import Rotation

import rospy
import rospkg

import geometry_msgs.msg
import mavros_msgs.msg

from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from mavros_msgs.msg import State

from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import SetMode

# import custom msg
from mavros_msgs.msg import ActuatorOutputsDRL

# import custom class
from quadrotor_dynamics import StateDifferentiator
from quadrotor_dynamics import ideal_model
from quadrotor_dynamics import ideal_model_v
from quadrotor_dynamics import get_lean_angle_from_quat

# update motion data when a new one is received
# motion_data = np.zeros([1, 14], dtype=float)
motion_data = np.zeros((14,), dtype=float)
def motion_callback(data):

    # declare global variable
    global motion_data

    # rospy.loginfo("I heard %s", type(data))
    # index = data.name.index('iris')
    index = data.name.index('if750a')
    # rospy.loginfo('states index of iris:%d', index)

    # header
    # model_state do not have time stamp
    motion_data[0] = rospy.Time.now().to_sec()

    # pose
    motion_pose = data.pose[index]
    motion_data[1] = motion_pose.position.x
    motion_data[2] = motion_pose.position.y
    motion_data[3] = motion_pose.position.z
    motion_data[4] = motion_pose.orientation.x
    motion_data[5] = motion_pose.orientation.y
    motion_data[6] = motion_pose.orientation.z
    motion_data[7] = motion_pose.orientation.w       # real part

    # twist
    motion_twist = data.twist[index]
    motion_data[8] = motion_twist.linear.x
    motion_data[9] = motion_twist.linear.y
    motion_data[10] = motion_twist.linear.z
    motion_data[11] = motion_twist.angular.x
    motion_data[12] = motion_twist.angular.y
    motion_data[13] = motion_twist.angular.z


# update state data when a new one is received
state_data = mavros_msgs.msg.State()
def state_callback(data):
    global state_data
    state_data = data


# init node
rospy.init_node('data_logger')

# declare subs, pubs and clients (default: infinite queue_size
# subs
rospy.Subscriber('/gazebo/model_states', ModelStates, motion_callback, queue_size=10)
rospy.Subscriber('mavros/state', State, state_callback, queue_size=10)

# pubs
# motion_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=1)
motor_pub = rospy.Publisher('/mavros/actuator_outputs_drl/actuator_sub', ActuatorOutputsDRL, queue_size=10)
pos_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)

# clients
# make sure that your service is available
rospy.loginfo("waiting for ROS services")
rospy.wait_for_service('mavros/cmd/arming')
rospy.wait_for_service('mavros/set_mode')
set_arm_client = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode)

# waiting for connecttion
rate_5 = rospy.Rate(5)
while not rospy.is_shutdown() and not state_data.connected:
    rate_5.sleep()
rospy.loginfo("connection success")

# send a few points
# switching to "OFFBOARD" mode is not allowed without sending points at first
rate_50 = rospy.Rate(50)
float_pose = geometry_msgs.msg.PoseStamped()
float_pose.pose.position.x = 0
float_pose.pose.position.y = 0
float_pose.pose.position.z = 1
for i in range(100):
    pos_pub.publish(float_pose)
    rate_50.sleep()
rospy.loginfo("send points success")

# set mode
while not rospy.is_shutdown() and state_data.mode != "OFFBOARD":
    set_mode_client(0, "OFFBOARD")
    # rospy.loginfo("Set Mode")
    rate_5.sleep()
rospy.loginfo("Set Mode success")

# set arm
while not rospy.is_shutdown() and state_data.armed != True:
    set_arm_client(True)
    rate_5.sleep()
rospy.loginfo("Arming success")

# takeoff
takeoff_time = 15
send_point_freq = 5
rate_send_point = rospy.Rate(send_point_freq)
float_pose.pose.position.z = 10     # float height from 1m to 10m
rospy.loginfo("TakeOff Start")
for i in range(takeoff_time*send_point_freq):
    if rospy.is_shutdown():
        break
    pos_pub.publish(float_pose)
    rate_send_point.sleep()
rospy.loginfo("TakeOff End")

# loiter control
loiter_time = 2
control_freq = 50
actuator_output = ActuatorOutputsDRL()
actuator_output.usedrl = True

# time.to_secs,pos*3,quaternion*4,linear*3,angular*3 # TODO(JC):log actions
# each row is a sample
motion_data_array = np.zeros([loiter_time*control_freq, 14], dtype=float)
motion_data_array = np.zeros([loiter_time*control_freq, 14], dtype=float)

rospy.loginfo("Loiter Control Start")
state_diff = StateDifferentiator()
state_diff.add_pose(motion_data[1:(3 + 1)], motion_data[4:(7 + 1)])     # init state (z0 is not zero)

for i in range(loiter_time*control_freq):
    if rospy.is_shutdown():
        break
    motion_data_array[i, :] = motion_data   # save data

    # state calculate
    state_diff.add_pose(motion_data[1:(3 + 1)], motion_data[4:(7 + 1)])

    # state -> action
    n_trajectory = 10
    n_length = 5
    pwm_ave = 1600
    pwm_sigma = 300     # pwm_ave - pwm_sigma < pwm < pwm_ave + pwm_sigma

    # TODO: replace for loop by tensor manipulation
    #  no need to change quat, nn use quat also
    pwm_normal = np.random.rand(n_trajectory, n_length, 4)  # PWM in (0,1)
    pwm_tensor = pwm_ave * np.ones((n_trajectory, n_length, 4)) + pwm_sigma * (2 * pwm_normal - np.ones((n_trajectory, n_length, 4)))
    # pwm_tensor = pwm_ave + pwm_sigma * (2 * pwm_normal - 1) # tensor broadcast

    spin2_tensor = np.square(pwm_tensor/10 - 90)
    k_f = 0.000806428
    k_tau = 1e-6
    lx = 0.265
    ly = 0.265
    lz = 0.35
    f_tensor = k_f * spin2_tensor.sum(axis=2)
    K_tau = np.array([k_f * ly * np.array([-1, 1, 1, -1]),
                      k_f * ly * np.array([-1, 1, 1, -1]),
                      k_tau * np.array([-1, -1, 1, 1])])
    tau_tensor = spin2_tensor.dot(K_tau.transpose())

    # # only print once
    # if i == 0:
    #     print('K_tau.shape', K_tau.shape)
    #     print('f_tensor.shape', f_tensor.shape)
    #     print('tau_tensor.shape', tau_tensor.shape)

    cost = np.zeros((n_trajectory,), dtype=float)
    p_now, q_now, vel_now, omega_now = state_diff.read_state()  # init state
    # TODO(JC): save state in traj searching ?
    # motion_mpc_array = np.zeros([n_trajectory * n_length, 14], dtype=float)
    for j in range(n_trajectory):

        p_traj = p_now
        q_traj = q_now
        vel_traj = vel_now
        omega_traj = omega_now
        for k in range(n_length):

            # # random act
            # pwm = pwm_tensor[j, k, :]
            # # generate acc
            # p_delta, q_delta, vel_delta, omega_delta = ideal_model(p_traj, q_traj, vel_traj, omega_traj, pwm)

            # random act
            # pwm = pwm_tensor[j, k, :]
            # generate acc
            p_delta, q_delta, vel_delta, omega_delta = ideal_model_v(p_traj, q_traj, vel_traj, omega_traj, f_tensor[j, k], tau_tensor[j, k, :])

            # next state
            p_traj = p_traj + p_delta
            q_traj = Rotation.from_quat(q_traj).__mul__(Rotation.from_quat(q_delta)).as_quat()
            vel_traj = vel_traj + vel_delta
            omega_traj = omega_traj + omega_delta
            # print('vel_traj',vel_traj,'omega_traj',omega_traj)
            # print('p_traj',p_traj,'q_traj',Rotation.from_quat(q_traj).as_euler('XYZ',degrees=True))

        # error func
        q_euler = get_lean_angle_from_quat(q_traj, degree=True)
        cost[j] = q_euler   # normalize to closed set [0,1] ?

    cost_min_index = np.argmin(cost)
    # TODO: zip this as a func fill_msg_with_numpy(act_out,pwm)
    actuator_output.output[0] = pwm_tensor[cost_min_index, 0, 0]
    actuator_output.output[1] = pwm_tensor[cost_min_index, 0, 1]
    actuator_output.output[2] = pwm_tensor[cost_min_index, 0, 2]
    actuator_output.output[3] = pwm_tensor[cost_min_index, 0, 3]
    motor_pub.publish(actuator_output)
    rate_50.sleep()     # TODO: deal with the problem that the first loop is not exactly 0.02s

# stop pwm override
actuator_output.usedrl = False
for i in range(10):
    if rospy.is_shutdown():
        break
    motor_pub.publish(actuator_output)
    rate_50.sleep()

# save data
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('motion_capture_fake')
save_time = datetime.now()
file_name = save_time.strftime('%y_%m_%d_%H_%M_%S')+'_loiter_ideal.txt'
# np.savetxt('Motion_data.txt', motion_data_array, fmt='%f')
np.savetxt(pkg_path + '/log/' + file_name, motion_data_array, fmt='%f')
rospy.loginfo("Loiter Control End")

# go back
go_back_time = 15
send_point_freq = 5
# rate_send_point = rospy.Rate(send_point_freq)
rospy.loginfo("GoBack Start")
for i in range(go_back_time*send_point_freq):
    if rospy.is_shutdown():
        break
    pos_pub.publish(float_pose)
    rate_send_point.sleep()
rospy.loginfo("Goback End")

# land
while not rospy.is_shutdown() and state_data.mode != "AUTO.LAND":
    set_mode_client(0, "AUTO.LAND")
    rate_5.sleep()
rospy.loginfo("LAND success")

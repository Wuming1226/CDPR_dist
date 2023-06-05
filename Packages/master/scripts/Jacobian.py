import numpy as np
import tf
import math


# Jacobian matrix of inverse kinematics

def getJacobian(AList, BList, position, orientation):
    """
    :param AList: list of positions of anchorAs on fixed frame (in world frame)
    :param Bpoint: list of positions of anchorBs on end effector (in body frame)
    :param position: position of end effector in world frame
    :param orientation: orientation of end effector in world frame
    :return: Jacobian matrix
    
    """

    transformedBList = []

    for i in range(0, len(BList)):
        # rotation
        # convert to quaternion form ("[x, y, z, 0]")
        vector = np.hstack([BList[i], 0])
        # rotation transform, convert body frame coordinates to world frame coordinates
        rotatedVector = tf.transformations.quaternion_multiply(orientation,
                            tf.transformations.quaternion_multiply(vector,
                            tf.transformations.quaternion_inverse(orientation)))
        # convert to 3D form
        rotatedB = np.delete(rotatedVector, 3)

        # translation
        transformedB = rotatedB + position

        transformedBList.append(transformedB)

    # calculate un and bn x un
    AList = np.array(AList)
    transformedBBList = np.array(transformedBList)
    uList = AList - transformedBList
    buList = AList - transformedBList
    for i in range(0, len(uList)):
        uList[i] = uList[i] / np.linalg.norm(uList[i])
    for i in range(0, len(uList)):
        buList[i] = np.cross(transformedBList[i], uList[i])

    
    # J =  - |    u1       u2    ...     un   | . T
    #        | b1 x u1  b2 x u2  ...  bn x un |
    #
    #           x  y  z    r  p  y
    #   =    |   u1.T    (b1 x u1).T  |
    #        |   u2.T    (b2 x u2).T  |
    #        |   ...         ...      |
    #        |   un.T    (bn x un).T  |
    #
    # un and bn are column vectors
    # uList and buList are row vectors, 
    J = np.row_stack((uList.T, 
                        buList.T))  
    J = -J.T

    # column 0,1,2 corespond to linear velocities, and 3,4,5 corespond to angular velocities
    J = J[:,0:3]
    return J



if __name__=="__main__":
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

    pos = np.array([0.273, 0.437, -0.352])
    orientation = np.array([0, 0, 0, 1])

    J = getJacobian(anchorAPos, anchorBPos, pos, orientation)

    velo = np.array([0.03, 0.01, 0.01])
    dx = np.matmul(J, velo.reshape(3, 1))
    dx = dx.reshape(1, 3)

    print(pos + dx)

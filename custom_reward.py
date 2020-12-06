import math
import numpy as np
from pybullet_utils import pd_controller_stable


def getRewardCustom(pose, humanoid):
    """Compute and return the pose-based reward."""
    pose_w = 0.65
    vel_w = 0.5
    end_eff_w = 0.15
    if humanoid._useComReward:
        com_w = 0.1
    else:
        com_w = 0

    total_w = pose_w + vel_w + end_eff_w + com_w
    pose_w /= total_w
    vel_w /= total_w
    end_eff_w /= total_w
    com_w /= total_w

    pose_scale = -2
    vel_scale = -0.1
    end_eff_scale = -40
    com_scale = -10

    reward = 0

    pose_err = 0
    vel_err = 0
    end_eff_err = 0
    com_err = 0

    if humanoid._useComReward:
        comSim, comSimVel = computeCOMposVel(humanoid, humanoid._sim_model)
        comKin, comKinVel = computeCOMposVel(humanoid, humanoid._kin_model)

    num_end_effs = 0
    num_joints = 15

    jointIndices = range(num_joints)
    simJointStates = humanoid._pybullet_client.getJointStatesMultiDof(humanoid._sim_model, jointIndices)
    kinJointStates = humanoid._pybullet_client.getJointStatesMultiDof(humanoid._kin_model, jointIndices)
    linkStatesSim = humanoid._pybullet_client.getLinkStates(humanoid._sim_model, jointIndices)
    linkStatesKin = humanoid._pybullet_client.getLinkStates(humanoid._kin_model, jointIndices)
    for j in range(num_joints):
        # TODO: r_p is the squared quaternion difference between orientations of reference model and actual model summed
        #           over all joints
        #       r_v is the squared difference between the angular velocities of the reference model and the actual model
        #           summed over all joints
        #       r_e is the squared difference between the 3D world position of the reference model and the actual model
        #           summed over all joints
        #       r_c is the squared difference between the center of mass of the reference model and the actual model
        #           summed over all joints

        curr_pose_err = 0
        curr_vel_err = 0

        # simJointInfo[0] -> position
        # simJointInfo[1] -> velocity
        # simJointInfo[2] -> joint force torque
        # simJointInfo[3] -> joint motor torque
        simJointInfo = simJointStates[j]
        kinJointInfo = kinJointStates[j]

        simJointPos = simJointInfo[0]
        kinJointPos = kinJointInfo[0]

        simJointVel = simJointInfo[1]
        kinJointVel = kinJointInfo[1]
        if len(simJointPos) == 1:
            # r_p is the squared difference between the target and the actual
            angle = simJointPos[0] - kinJointPos[0]
            curr_pose_err = angle * angle
            # r_v is the squared difference between the target and the actual
            velocity_difference = simJointVel[0] - kinJointVel[0]
            curr_vel_err = velocity_difference * velocity_difference
        if len(simJointPos) == 4:
            diffQuat = humanoid._pybullet_client.getDifferenceQuaternion(simJointPos, kinJointPos)
            # r_p is the squared difference between the target and the actual
            _, angle = humanoid._pybullet_client.getAxisAngleFromQuaternion(diffQuat)
            curr_pose_err = angle * angle
            # r_v is the squared difference between the target and the actual
            velocity_differences = [
                simJointVel[0] - kinJointVel[0], simJointVel[1] - kinJointVel[1],
                simJointVel[2] - kinJointVel[2]
            ]
            curr_vel_err = velocity_differences[0] * velocity_differences[0] + \
                velocity_differences[1] * velocity_differences[1] + \
                velocity_differences[2] * velocity_differences[2]

        pose_err += curr_pose_err
        vel_err += curr_vel_err

        if j in humanoid._end_effectors:  # for effector reward
            linkStateSim = linkStatesSim[j]
            linkStateKin = linkStatesKin[j]

            simLinkStatePos = linkStateSim[0]
            kinLinkStatePos = linkStateKin[0]

            posDiffs = [simLinkStatePos[x] - kinLinkStatePos[x] for x in range(len(simLinkStatePos))]

            curr_end_err = posDiffs[0] * posDiffs[0] + \
                posDiffs[1] * posDiffs[1] + posDiffs[2] * posDiffs[2]

            end_eff_err += curr_end_err
            num_end_effs += 1

    if num_end_effs > 0:
        end_eff_err /= num_end_effs

    # COM error in initial code -> COM velocities
    if humanoid._useComReward:
        com_err = 0.1 * np.sum(np.square(comKinVel - comSimVel))

    pose_reward = math.exp(pose_scale * pose_err)
    vel_reward = math.exp(vel_scale * vel_err)
    end_eff_reward = math.exp(end_eff_scale * end_eff_err)
    com_reward = math.exp(com_scale * com_err)

    reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward + \
        com_w * com_reward

    return reward


def computeCOMposVel(humanoid, uid: int):
    """Compute center-of-mass position and velocity."""
    pb = humanoid._pybullet_client
    num_joints = 15
    jointIndices = range(num_joints)
    link_states = pb.getLinkStates(uid, jointIndices, computeLinkVelocity=1)
    link_pos = np.array([s[0] for s in link_states])
    link_vel = np.array([s[-2] for s in link_states])
    tot_mass = 0.
    masses = []
    for j in jointIndices:
        mass_, *_ = pb.getDynamicsInfo(uid, j)
        masses.append(mass_)
        tot_mass += mass_
    masses = np.asarray(masses)[:, None]
    com_pos = np.sum(masses * link_pos, axis=0) / tot_mass
    com_vel = np.sum(masses * link_vel, axis=0) / tot_mass
    return com_pos, com_vel


def quatMul(q1, q2):
    return [
        q1[3] * q2[0] + q2[3] * q1[0] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] + q2[3] * q1[1] + q1[2] * q2[0] - q1[0] * q2[2],
        q1[3] * q2[2] + q2[3] * q1[2] + q1[0] * q2[1] - q1[1] * q2[0],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q1[2]
    ]


def calcRootAngVelErr(vel0, vel1):
    diff = [vel0[0] - vel1[0], vel0[1] - vel1[1], vel0[2] - vel1[2]]
    return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]


def calcRootRotDiff(humanoid, orn0, orn1):
    # conjugate is found by negating the input
    conjugate = [-x for x in orn0]
    q_diff = quatMul(orn1, conjugate)
    _, angle = humanoid._pybullet_client.getAxisAngleFromQuaternion(q_diff)
    return angle * angle

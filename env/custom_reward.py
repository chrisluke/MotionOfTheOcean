import math
import numpy as np
from pybullet_utils import pd_controller_stable

def getRewardCustom(pose, humanoid):
    """Compute and return the pose-based reward."""
    #from DeepMimic double cSceneImitate::CalcRewardImitate
    #todo: compensate for ground height in some parts, once we move to non-flat terrain
    # not values from the paper, but from the published code.
    pose_w = 0.5
    vel_w = 0.05
    end_eff_w = 0.15
    # does not exist in paper
    root_w = 0.2
    if humanoid._useComReward:
      com_w = 0.1
    else:
      com_w = 0

    total_w = pose_w + vel_w + end_eff_w + root_w + com_w
    pose_w /= total_w
    vel_w /= total_w
    end_eff_w /= total_w
    root_w /= total_w
    com_w /= total_w

    pose_scale = 2
    vel_scale = 0.1
    end_eff_scale = 40
    root_scale = 5
    com_scale = 10
    err_scale = 1  # error scale

    reward = 0

    pose_err = 0
    vel_err = 0
    end_eff_err = 0
    root_err = 0
    com_err = 0
    heading_err = 0

    
    if humanoid._useComReward:
      comSim, comSimVel = computeCOMposVel(humanoid,humanoid._sim_model)
      comKin, comKinVel = computeCOMposVel(humanoid,humanoid._kin_model)
    
    root_id = 0
    

    mJointWeights = [
        0.20833, 0.10416, 0.0625, 0.10416, 0.0625, 0.041666666666666671, 0.0625, 0.0416, 0.00,
        0.10416, 0.0625, 0.0416, 0.0625, 0.0416, 0.0000
    ]

    num_end_effs = 0
    num_joints = 15

    root_rot_w = mJointWeights[root_id]
    rootPosSim, rootOrnSim = humanoid._pybullet_client.getBasePositionAndOrientation(humanoid._sim_model)
    rootPosKin, rootOrnKin = humanoid._pybullet_client.getBasePositionAndOrientation(humanoid._kin_model)
    linVelSim, angVelSim = humanoid._pybullet_client.getBaseVelocity(humanoid._sim_model)
    #don't read the velocities from the kinematic model (they are zero), use the pose interpolator velocity
    #see also issue https://github.com/bulletphysics/bullet3/issues/2401 
    linVelKin = humanoid._poseInterpolator._baseLinVel
    angVelKin = humanoid._poseInterpolator._baseAngVel

    root_rot_err = calcRootRotDiff(humanoid,rootOrnSim, rootOrnKin)
    pose_err += root_rot_w * root_rot_err

    root_vel_diff = [
        linVelSim[0] - linVelKin[0], linVelSim[1] - linVelKin[1], linVelSim[2] - linVelKin[2]
    ]
    root_vel_err = root_vel_diff[0] * root_vel_diff[0] + root_vel_diff[1] * root_vel_diff[
        1] + root_vel_diff[2] * root_vel_diff[2]

    root_ang_vel_err = calcRootAngVelErr(angVelSim, angVelKin)
    vel_err += root_rot_w * root_ang_vel_err

    useArray = True
    
    if useArray:
      jointIndices = range(num_joints)
      simJointStates = humanoid._pybullet_client.getJointStatesMultiDof(humanoid._sim_model, jointIndices)
      kinJointStates = humanoid._pybullet_client.getJointStatesMultiDof(humanoid._kin_model, jointIndices)
    if useArray:
      linkStatesSim = humanoid._pybullet_client.getLinkStates(humanoid._sim_model, jointIndices)
      linkStatesKin = humanoid._pybullet_client.getLinkStates(humanoid._kin_model, jointIndices)
    for j in range(num_joints):
      curr_pose_err = 0
      curr_vel_err = 0
      w = mJointWeights[j]
      if useArray:
        simJointInfo = simJointStates[j]
      else:
        simJointInfo = humanoid._pybullet_client.getJointStateMultiDof(humanoid._sim_model, j)

      #print("simJointInfo.pos=",simJointInfo[0])
      #print("simJointInfo.vel=",simJointInfo[1])
      if useArray:
        kinJointInfo = kinJointStates[j]
      else:
        kinJointInfo = humanoid._pybullet_client.getJointStateMultiDof(humanoid._kin_model, j)
      #print("kinJointInfo.pos=",kinJointInfo[0])
      #print("kinJointInfo.vel=",kinJointInfo[1])
      if (len(simJointInfo[0]) == 1):
        angle = simJointInfo[0][0] - kinJointInfo[0][0]
        curr_pose_err = angle * angle
        velDiff = simJointInfo[1][0] - kinJointInfo[1][0]
        curr_vel_err = velDiff * velDiff
      if (len(simJointInfo[0]) == 4):
        #print("quaternion diff")
        diffQuat = humanoid._pybullet_client.getDifferenceQuaternion(simJointInfo[0], kinJointInfo[0])
        axis, angle = humanoid._pybullet_client.getAxisAngleFromQuaternion(diffQuat)
        curr_pose_err = angle * angle
        diffVel = [
            simJointInfo[1][0] - kinJointInfo[1][0], simJointInfo[1][1] - kinJointInfo[1][1],
            simJointInfo[1][2] - kinJointInfo[1][2]
        ]
        curr_vel_err = diffVel[0] * diffVel[0] + diffVel[1] * diffVel[1] + diffVel[2] * diffVel[2]

      pose_err += w * curr_pose_err
      vel_err += w * curr_vel_err

      is_end_eff = j in humanoid._end_effectors
      
      if is_end_eff:

        if useArray:
          linkStateSim = linkStatesSim[j]
          linkStateKin = linkStatesKin[j]
        else:
          linkStateSim = humanoid._pybullet_client.getLinkState(humanoid._sim_model, j)
          linkStateKin = humanoid._pybullet_client.getLinkState(humanoid._kin_model, j)
        linkPosSim = linkStateSim[0]
        linkPosKin = linkStateKin[0]
        linkPosDiff = [
            linkPosSim[0] - linkPosKin[0], linkPosSim[1] - linkPosKin[1],
            linkPosSim[2] - linkPosKin[2]
        ]
        curr_end_err = linkPosDiff[0] * linkPosDiff[0] + linkPosDiff[1] * linkPosDiff[
            1] + linkPosDiff[2] * linkPosDiff[2]
        end_eff_err += curr_end_err
        num_end_effs += 1

    if (num_end_effs > 0):
      end_eff_err /= num_end_effs

    #double root_ground_h0 = mGround->SampleHeight(sim_char.GetRootPos())
    #double root_ground_h1 = kin_char.GetOriginPos()[1]
    #root_pos0[1] -= root_ground_h0
    #root_pos1[1] -= root_ground_h1
    root_pos_diff = [
        rootPosSim[0] - rootPosKin[0], rootPosSim[1] - rootPosKin[1], rootPosSim[2] - rootPosKin[2]
    ]
    root_pos_err = root_pos_diff[0] * root_pos_diff[0] + root_pos_diff[1] * root_pos_diff[
        1] + root_pos_diff[2] * root_pos_diff[2]
    #
    #root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1)
    #root_rot_err *= root_rot_err

    #root_vel_err = (root_vel1 - root_vel0).squaredNorm()
    #root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm()

    root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err

    # COM error in initial code -> COM velocities
    if humanoid._useComReward:
      com_err = 0.1 * np.sum(np.square(comKinVel - comSimVel))
    # com_err = 0.1 * np.sum(np.square(comKin - comSim))
    #com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm()

    #print("pose_err=",pose_err)
    #print("vel_err=",vel_err)
    pose_reward = math.exp(-err_scale * pose_scale * pose_err)
    vel_reward = math.exp(-err_scale * vel_scale * vel_err)
    end_eff_reward = math.exp(-err_scale * end_eff_scale * end_eff_err)
    root_reward = math.exp(-err_scale * root_scale * root_err)
    com_reward = math.exp(-err_scale * com_scale * com_err)

    reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward + root_w * root_reward + com_w * com_reward
    
    info_rew = dict(
      pose_reward=pose_reward,
      vel_reward=vel_reward,
      end_eff_reward=end_eff_reward,
      root_reward=root_reward,
      com_reward=com_reward
    )
    
    info_errs = dict(
      pose_err=pose_err,
      vel_err=vel_err,
      end_eff_err=end_eff_err,
      root_err=root_err,
      com_err=com_err
    )
    
    return reward

def computeCOMposVel(humanoid,uid: int):
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
        q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2],
        q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
    ]

def calcRootAngVelErr(vel0, vel1):
    diff = [vel0[0] - vel1[0], vel0[1] - vel1[1], vel0[2] - vel1[2]]
    return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

def calcRootRotDiff(humanoid, orn0, orn1):
    orn0Conj = [-orn0[0], -orn0[1], -orn0[2], orn0[3]]
    q_diff = quatMul(orn1, orn0Conj)
    axis, angle = humanoid._pybullet_client.getAxisAngleFromQuaternion(q_diff)
    return angle * angle
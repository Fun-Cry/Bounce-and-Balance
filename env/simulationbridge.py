"""
Copyright (C) 2020 Benjamin Bokser
"""

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
import os

from env import actuator
from env import actuator_param
from env.utils import L, Z, Q_inv
from env.scenes import create_staircase
useRealTime = 1  # Do NOT change to real time


def reaction(numJoints, bot):  # returns joint reaction force
    reaction = np.array([j[2] for j in p.getJointStates(bot, range(numJoints))])  # 4x6 array [Fx, Fy, Fz, Mx, My, Mz]
    forces = reaction[:, 0:3]  # selected all joints [Fx, Fy, Fz]
    torques = reaction[:, 5]
    return forces, torques  # f = np.linalg.norm(reaction[:, 0:3], axis=1)  # magnitude of F


class Sim:

    def __init__(self, X_0, model, spring, q_cal, dt, mu, g=9.807, fixed=False,
                 record=False, scale=1, gravoff=False, direct=False):
        self.q_cal = q_cal
        self.dt = dt
        self.record_rt = record  # record video in real time
        self.L = model["linklengths"]
        self.model = model["model"]
        self.n_a = model["n_a"]
        self.S = model["S"]
        self.spring_fn = spring.fn_spring
        self.actuator_q0 = actuator.Actuator(dt=dt, model=actuator_param.actuator_rmdx10)
        self.actuator_q2 = actuator.Actuator(dt=dt, model=actuator_param.actuator_rmdx10)
        self.actuator_rw1 = actuator.Actuator(dt=dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rw2 = actuator.Actuator(dt=dt, model=actuator_param.actuator_r100kv90)
        self.actuator_rwz = actuator.Actuator(dt=dt, model=actuator_param.actuator_8318)  # r80kv110
        self.f_sens = None
        self.tau_sens = None
        
        if gravoff == True:
            GRAVITY = 0
        else:
            GRAVITY = -g

        if direct is True:
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        self.plane = p.loadURDF("plane.urdf")
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.path_parent = os.getcwd()
        model_path = model["urdfpath"]
        # self.bot = p.loadURDF(os.path.join(path_parent, os.path.pardir, model_path), [0, 0, 0.7 * scale],
        self.bot = p.loadURDF(os.path.join(self.path_parent, model_path), X_0[0:3],  # 0.31
                         robotStartOrientation, useFixedBase=fixed, globalScaling=scale,
                         flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-45, cameraTargetPosition=[0,0,0])
        self.jointArray = range(p.getNumJoints(self.bot))
        p.setGravity(0, 0, GRAVITY)
        p.setTimeStep(self.dt)
        self.numJoints = p.getNumJoints(self.bot)

        p.setRealTimeSimulation(useRealTime)

        # p.createConstraint(self.bot, 3, -1, -1, p.JOINT_POINT2POINT, [0, 0, 0], [-0.135, 0, 0], [0, 0, 0])
        jconn_1 = [x * scale for x in [0.135, 0, 0]]
        jconn_2 = [x * scale for x in [-0.0014381, 0, 0.01485326948]]
        
        print(jconn_1, jconn_2)
        linkjoint = p.createConstraint(self.bot, 1, self.bot, 3, p.JOINT_POINT2POINT, [0, 0, 0], jconn_1, jconn_2)
        p.changeConstraint(linkjoint, maxForce=1000)
        self.c_link = 3

        # increase friction of toe to ideal
        # p.changeDynamics(self.bot, self.c_link, lateralFriction=2, contactStiffness=100000, contactDamping=10000)
        p.changeDynamics(self.bot, self.c_link, lateralFriction=mu)  # , restitution=0.01)

        for i in range(self.numJoints):
            # Disable the default velocity/position motor:
            # force=1 allows us to easily mimic joint friction rather than disabling
            p.setJointMotorControl2(self.bot, i, p.VELOCITY_CONTROL, force=0)  # force=0.5
            # enable joint torque sensing
            p.enableJointForceTorqueSensor(self.bot, i, 1)
            # increase max joint velocity (default = 100 rad/s)
            p.changeDynamics(self.bot, i, maxJointVelocity=800)  # max 3800 rpm

        p.resetBaseVelocity(self.bot, X_0[7:10], X_0[10:13])  # give initial lin & angular velocity (if there is any)
        self.X = np.zeros(13)  # initialize state
        self.init = True
        self.Q_calib = np.array([1, 0, 0, 0])
        self.i = 0
        self.ii = 0

    def sim_run(self, u):
        # import pybullet as p

        # Assume robot_id is your robot's unique id returned by p.loadURDF() (or you already have it)
        robot_id = self.bot  # Use your existing robot id

        num_joints = p.getNumJoints(robot_id)
        # print("Number of joints:", num_joints)

        joint_index_mapping = {}  # dictionary to hold name:index mapping

        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('UTF-8')
            joint_index_mapping[joint_name] = i
            # print(f"Index: {i}, Joint Name: {joint_name}")

        # Now you know which index corresponds to which joint (e.g., joint_0, joint_rw0, etc.)

        
        
        q = np.array([j[0] for j in p.getJointStates(self.bot, range(0, self.numJoints))])
        dq = np.array([j[1] for j in p.getJointStates(self.bot, range(0, self.numJoints))])
        qa = (q.T @ self.S).flatten()
        dqa = (dq.T @ self.S).flatten()

        q0 = qa[0] + self.q_cal[0]
        q2 = qa[1] + self.q_cal[1]
        tau_s = self.spring_fn(q0=q0, q2=q2)

        tau = np.zeros(self.n_a)
        i = np.zeros(self.n_a)
        v = np.zeros(self.n_a)

        u *= -1
        # tau[0], i[0], v[0] = self.actuator_q0.actuate(i=u[0], q_dot=dqa[0]) + tau_s[0]
        # tau[1], i[1], v[1] = self.actuator_q2.actuate(i=u[1], q_dot=dqa[1]) + tau_s[1]
        τ0, i0, v0 = self.actuator_q0.actuate(i=u[0], q_dot=dqa[0])
        tau[0] = τ0 + tau_s[0]
        i[0] = i0
        v[0] = v0

        τ2, i2, v2 = self.actuator_q2.actuate(i=u[1], q_dot=dqa[1])
        tau[1] = τ2 + tau_s[1]
        i[1] = i2
        v[1] = v2

        
        tau[2], i[2], v[2] = self.actuator_rw1.actuate(i=u[2], q_dot=dqa[2])
        tau[3], i[3], v[3] = self.actuator_rw2.actuate(i=u[3], q_dot=dqa[3])
        tau[4], i[4], v[4] = self.actuator_rwz.actuate(i=u[4], q_dot=dqa[4])

        # tau[4] *= 0
        torque = self.S @ tau

        p.setJointMotorControlArray(self.bot, self.jointArray, p.TORQUE_CONTROL, forces=torque)
        p_base, Q_base_p = p.getBasePositionAndOrientation(self.bot)
        Q_base = np.roll(Q_base_p, 1)  # pybullet gives quaternions in xyzw format instead of wxyz, shift values.
        if self.init is True:
            """
            For some unknown reason, PyBullet doesn't always start the sim with Q = [1, 0, 0, 0] even though the body
            is loaded in with that attitude.
            Set the starting Q as the "calibration" quaternion on first timestep to rotate Q back to the measurement
            it SHOULD be returning.
            """
            self.Q_calib = Q_base
            self.init = False
        Q_base = L(self.Q_calib).T @ Q_base  # correct Q_base by rotating it by Q_calib
        velocities = p.getBaseVelocity(self.bot)

        self.X[0:3] = p_base
        self.X[3:7] = Q_base
        self.X[7:10] = Z(Q_inv(Q_base), velocities[0])  # linear vel world -> body frame
        self.X[10:] = Z(Q_inv(Q_base), velocities[1])  # angular vel world -> body frame

        self.f_sens, self.tau_sens = reaction(self.numJoints, self.bot)
        contact = np.array(p.getContactPoints(self.bot, self.plane, self.c_link), dtype=object)
        if np.shape(contact)[0] == 0:  # prevent empty list from being passed
            grf = np.zeros(3)
            c = False
        else:
            grf_nrml_onB = np.array(contact[0, 7])
            grf_nrml = contact[0, 9]
            fric1 = contact[0, 10]
            fric1_dir = np.array(contact[0, 11])
            fric2 = contact[0, 12]
            fric2_dir = np.array(contact[0, 13])
            grf_z = grf_nrml * grf_nrml_onB
            fric_y = fric1 * fric1_dir
            fric_x = fric2 * fric2_dir
            grf = (grf_z - fric_y - fric_x).flatten()
            c = True  # Detect contact with ground plane

        # Record Video in real time
        self.ii += 1
        if self.record_rt is True and self.ii == 24:  # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "file1.mp4")
            self.ii = 0
            self.i += 1
            # fix camera onto model
            p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=50, cameraPitch=-20, cameraTargetPosition=p_base)
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(640, 480, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            im = Image.fromarray(rgbImg)
            im.save(self.path_parent + '/imgs/' + str(self.i).zfill(4) + ".png")
            """To convert these images to video, run the following command in /imgs:
            cat *.png | ffmpeg -f image2pipe -i - output.mp4
            """

        if useRealTime == 0:
            p.stepSimulation()

        return self.X, qa, dqa, c, tau, i, v, 
    
    def get_camera_image(self, link_index=0, width=128, height=128, fov=90, near=0.01, far=5.0):
        """
        Renders an RGB image from the perspective of the given link (default = base link).
        """
        # Get the pose of the link in the world frame
        pos, orn = p.getLinkState(self.bot, link_index)[:2]

        # Get view and projection matrices
        rot_matrix = p.getMatrixFromQuaternion(orn)
        forward_vec = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        up_vec = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
        cam_target = np.array(pos) + 0.5 * forward_vec

        view_matrix = p.computeViewMatrix(cameraEyePosition=pos,
                                        cameraTargetPosition=cam_target,
                                        cameraUpVector=up_vec)

        aspect = width / height
        proj_matrix = p.computeProjectionMatrixFOV(fov=fov,
                                                aspect=aspect,
                                                nearVal=near,
                                                farVal=far)

        # Get camera image
        img = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
        rgb_array = np.reshape(img[2], (height, width, 4))[:, :, :3]  # strip alpha
        return rgb_array
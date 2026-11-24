import mink
import mujoco
import numpy as np

from pathlib import Path

class IK_Solver:
    def __init__(
        self,
        scene_xml,
        active_joints,
        num_motors,
        ee_body,
        ee_offset,
        ee_rot=np.eye(3),
        base_body="base",
        max_joint_vel=1.0,
        position_cost=1.0,
        orientation_cost=0.5,
        posture_cost=1e-2,
        damping=1e-3,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(Path(scene_xml)))
        self.configuration = mink.Configuration(self.model)
        self.gravity_data = mujoco.MjData(self.model)

        motor_joints = self.model.actuator_trnid[:num_motors, 0]
        self.motor_qpos = self.model.jnt_qposadr[motor_joints]
        self.motor_dof = self.model.jnt_dofadr[motor_joints]

        self.ee_offset = np.asarray(ee_offset, dtype=float)
        self.ee_rot = np.asarray(ee_rot, dtype=float)
        self.target_pos = None

        self.hand_task = mink.FrameTask(ee_body, "body", position_cost=position_cost, orientation_cost=orientation_cost)
        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)
        self.base_task = mink.FrameTask(base_body, "body", position_cost=1e3, orientation_cost=1e3)
        self.tasks = [self.hand_task, self.posture_task, self.base_task]

        vel = {}
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = self.model.joint(j).name
            vel[name] = max_joint_vel if name in active_joints else 0.0
        self.limits = [mink.ConfigurationLimit(self.model), mink.VelocityLimit(self.model, vel)]

        self.damping = damping

    def sync(self, q_motor):
        q = self.configuration.q.copy()
        q[self.motor_qpos] = q_motor
        # keep every limited 1-DoF joint inside its range (e.g. gripper joints start at 0 < lower limit 0.01)
        for j in range(self.model.njnt):
            if self.model.jnt_limited[j] and self.model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                adr = self.model.jnt_qposadr[j]
                q[adr] = np.clip(q[adr], *self.model.jnt_range[j])
        self.configuration.update(q)
        self.posture_task.set_target(self.configuration.q)
        self.base_task.set_target_from_configuration(self.configuration)

    def set_target(self, pos):
        self.target_pos = np.asarray(pos, dtype=float)
        body_pos = self.target_pos - self.ee_rot @ self.ee_offset
        target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(self.ee_rot), body_pos)
        self.hand_task.set_target(target)

    def step(self, dt):
        vel = mink.solve_ik(self.configuration, self.tasks, dt, solver="daqp", damping=self.damping, limits=self.limits)
        self.configuration.integrate_inplace(vel, dt)
        return self.configuration.q[self.motor_qpos].copy()

    def reached(self, tol=1e-3):
        error = self.hand_task.compute_error(self.configuration)
        return np.linalg.norm(error[:3]) < tol

    def gravity_torque(self, q_motor):
        self.gravity_data.qpos[:] = self.configuration.q
        self.gravity_data.qpos[self.motor_qpos] = q_motor
        self.gravity_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.gravity_data)
        return self.gravity_data.qfrc_bias[self.motor_dof].copy()
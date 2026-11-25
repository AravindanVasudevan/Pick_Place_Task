import argparse
parser = argparse.ArgumentParser(description="Scripted pick and place demonstrations collection over DDS")
parser.add_argument("--episodes", type=int, default=5, help="number of episodes to record")
args = parser.parse_args()

import pickle
import sys, time
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pndbotics_sdk_py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from pndbotics_sdk_py.idl.default import pnd_adam_msg_dds__LowCmd_, pnd_adam_msg_dds__HandCmd_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowCmd_, LowState_, HandCmd_
from msg.camera_msg import CameraImage
from msg.reset_msg import SimResetSignal

from utils.ik_solver import IK_Solver
from utils.timing import CONTROL_DT, RECORD_EVERY, wait_for_next_tick

NUM_MOTOR = 19

KP_CONFIG = [
    405.0,  # waistRoll (0)
    405.0,  # waistPitch (1)
    205.0,  # waistYaw (2)
    9.0,   # neckYaw (3)
    9.0,   # neckPitch (4)
    180.0,  # shoulderPitch_Left (5)
    180.0,   # shoulderRoll_Left (6)
    50.0,   # shoulderYaw_Left (7) raised from 9
    50.0,   # elbow_Left (8) raised from 9
    50.0,   # wristYaw_Left (9) raised from 9
    50.0,   # wristPitch_Left (10) raised from 9
    50.0,   # wristRoll_Left (11) raised from 9
    180.0,  # shoulderPitch_Right (12)
    180.0,  # shoulderRoll_Right (13)
    50.0,   # shoulderYaw_Right (14) raised from 9
    50.0,  # elbow_Right (15) raised from 9
    50.0,  # wristYaw_Right (16) raised from 9
    50.0,  # wristPitch_Right (17) raised from 9
    50.0   # wristRoll_Right (18) raised from 9
]

KD_CONFIG = [
    6.75,   # waistRoll (0)
    6.75,   # waistPitch (1)
    3.42,   # waistYaw (2)
    0.9,   # neckYaw (3)
    0.9,   # neckPitch (4)
    1.8,   # shoulderPitch_Left (5)
    1.8,   # shoulderRoll_Left (6)
    2.5,   # shoulderYaw_Left (7) raised from 0.9
    2.5,   # elbow_Left (8) raised from 0.9
    2.5,   # wristYaw_Left (9) raised from 0.9
    2.5,   # wristPitch_Left (10) raised from 0.9
    2.5,   # wristRoll_Left (11) raised from 0.9
    1.8,   # shoulderPitch_Right (12)
    1.8,   # shoulderRoll_Right (13)
    2.5,   # shoulderYaw_Right (14) raised from 0.9
    2.5,   # elbow_Right (15) raised from 0.9
    2.5,   # wristYaw_Right (16) raised from 0.9
    2.5,   # wristPitch_Right (17) raised from 0.9
    2.5    # wristRoll_Right (18) raised from 0.9
]

HOME_POS = np.array([0, 0, 0,
                     0, 0.5,
                     0, 0, 0, 0,
                     0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0], dtype=float)

# Paths
SCENE_XML = ROOT / "task_imitation_learning" / "scene" / "pick_place_scene.xml"
DATA_DIR = ROOT / "task_imitation_learning" / "data"
TASK_PROMPT = "Pick up the nut and place it on the green target."

# Values experimented with ######################################################################
PINCH_OPEN = [0, 0, 0, 400, 600, 1000]
PINCH_CLOSED = [0, 0, 0, 400, 300, 1000]
HAND_HOME_POS = np.array(PINCH_OPEN + PINCH_OPEN, dtype=int)
HAND_SLICE = {"R": slice(6, 12), "L": slice(0, 6)}

ARMS = {
    "R": dict(
            joints=["waistPitch", "waistYaw", "shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right",
                    "elbow_Right", "wristYaw_Right", "wristPitch_Right", "wristRoll_Right"],
            wrist="wristRollRight",
            grasp_offset=[0.0509, 0.0126, -0.1271],
            fingers_down=np.array([[-0.9934, -0.0922, -0.0687],
                                   [ 0.0924, -0.9957,  0.0   ],
                                   [-0.0684, -0.0063,  0.9976]])),
    "L": dict(
            joints=["waistPitch", "waistYaw", "shoulderPitch_Left", "shoulderRoll_Left", "shoulderYaw_Left",
                    "elbow_Left", "wristYaw_Left", "wristPitch_Left", "wristRoll_Left"],
            wrist="wristRollLeft",
            grasp_offset=[0.0509, -0.0122, -0.1271],
            fingers_down=np.array([[-0.9960,  0.0854, -0.0251],
                                   [-0.0855, -0.9963,  0.0   ],
                                   [-0.0250,  0.0021,  0.9997]])),
}

NUT_Z = 1.035
NUT_CENTRE_OFFSET_X = 0.008
APPROACH_HEIGHT = 0.15
GRASP_HEIGHT = 0.0 #-0.006
PLACE_HEIGHT = 0.005
LIFT_HEIGHT = 0.10
TRAVEL_TOL = 0.003


ARM_OUT_POS = HOME_POS.copy()
ARM_OUT_POS[13] = -0.8
ELBOW_BENT_POS = ARM_OUT_POS.copy()
ELBOW_BENT_POS[15] = -1.6
ARM_FORWARD_POS = HOME_POS.copy()
ARM_FORWARD_POS[[12, 13, 15, 17]] = [-0.3, -0.8, -1.2, 0.9]
ABOVE_TABLE_POS = HOME_POS.copy()
ABOVE_TABLE_POS[[12, 15, 17]] = [-0.3, -1.2, 0.9]
RAISE_ARM_ROUTE = [ARM_OUT_POS, ELBOW_BENT_POS, ARM_FORWARD_POS, ABOVE_TABLE_POS]
LOWER_ARM_ROUTE = [ARM_FORWARD_POS, ELBOW_BENT_POS, ARM_OUT_POS, HOME_POS]

MIRROR_SIGN = np.array([1, -1, -1, 1, -1, 1, -1])

def mirror_to_left(pos):
    left = HOME_POS.copy()
    left[0:5] = pos[0:5]
    left[5:12] = pos[12:19] * MIRROR_SIGN
    return left

ROUTES = {"R": dict(raise_arm=RAISE_ARM_ROUTE, lower_arm=LOWER_ARM_ROUTE, above_table=ABOVE_TABLE_POS),
          "L": dict(raise_arm=[mirror_to_left(p) for p in RAISE_ARM_ROUTE],
                    lower_arm=[mirror_to_left(p) for p in LOWER_ARM_ROUTE],
                    above_table=mirror_to_left(ABOVE_TABLE_POS))}

IK_SPEED = 0.04

# Values experimented with ######################################################################

IK_SOLVERS = {arm: IK_Solver(SCENE_XML,
                             active_joints=cfg["joints"],
                             num_motors=NUM_MOTOR,
                             ee_body=cfg["wrist"],
                             ee_offset=cfg["grasp_offset"],
                             ee_rot=cfg["fingers_down"],
                             base_body="lifting_Columns",
                             max_joint_vel=0.25)
            for arm, cfg in ARMS.items()
}


ChannelFactoryInitialize(1, "lo")

cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
cmd_pub.Init()
cmd = pnd_adam_msg_dds__LowCmd_(NUM_MOTOR)

hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
hand_pub.Init()
hand_cmd = pnd_adam_msg_dds__HandCmd_()

cur_pos_sub = ChannelSubscriber("rt/lowstate", LowState_)
cur_pos_sub.Init()

cam_sub = ChannelSubscriber("rt/camera/image", CameraImage)
cam_sub.Init()

reset_done_sub = ChannelSubscriber("rt/sim/reset_done", SimResetSignal)
reset_done_sub.Init()

reset_request_pub = ChannelPublisher("rt/sim/reset_request", SimResetSignal)
reset_request_pub.Init()



def send_cmd(motor_pos, hand_pos=HAND_HOME_POS):
    gravity_tau = IK_SOLVERS["R"].gravity_torque(motor_pos)

    for i in range(NUM_MOTOR):
        cmd.motor_cmd[i].q = float(motor_pos[i])
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kp = KP_CONFIG[i]
        cmd.motor_cmd[i].kd = KD_CONFIG[i]
        cmd.motor_cmd[i].tau = float(gravity_tau[i])

    for i in range(len(hand_pos)):
        hand_cmd.position[i] = int(hand_pos[i])

    cmd_pub.Write(cmd)
    hand_pub.Write(hand_cmd)

def read_pos():
    cur_pos = cur_pos_sub.Read(timeout=1.0)

    if cur_pos is None:
        raise RuntimeError("rt/lowstate not running check scene.py")
    
    return np.array([cur_pos.motor_state[i].q for i in range(NUM_MOTOR)])

def go_home(duration=3.0):
    pos_0 = read_pos()
    t0 = time.perf_counter()

    while (t:= time.perf_counter() - t0) < duration:
        phase = np.tanh(t/ 1.2)
        pos_to_send = (1-phase) * pos_0 + (phase * HOME_POS)
        send_cmd(pos_to_send, HAND_HOME_POS)
        time.sleep(CONTROL_DT)

def hold_home(duration):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        send_cmd(HOME_POS, HAND_HOME_POS)
        time.sleep(CONTROL_DT)

def reset_sim(ep, timeout=3.0):
    """Ask scene.py to randomize objects; block until it confirms."""
    reset_request_pub.Write(SimResetSignal(episode_id=ep, nut_pos=[0.0] * 3, target_pos=[0.0] * 3,
                                           base_pos=[0.0] * 3, base_quat=[1.0, 0.0, 0.0, 0.0], success=False), timeout=1.0)
    t0 = time.time()
    while time.time() - t0 < timeout:
        send_cmd(HOME_POS, HAND_HOME_POS)
        msg = reset_done_sub.Read(timeout=0.05)
        if msg is not None and msg.episode_id == ep:
            set_ik_base(msg.base_pos, msg.base_quat)
            return np.array(msg.nut_pos), np.array(msg.target_pos), msg.success
    raise TimeoutError(f"No reset_done for episode {ep}")


def set_ik_base(base_pos, base_quat):
    """Copy the simulator's settled floating-base pose into the IK model."""
    for ik in IK_SOLVERS.values():
        q = ik.configuration.q.copy()
        q[0:3] = base_pos
        q[3:7] = base_quat
        ik.configuration.update(q)


def get_hand_pos(grasp, arm):
    """Boolean hand True = hand closed False = open."""
    hand_pos = HAND_HOME_POS.copy()
    hand_pos[HAND_SLICE[arm]] = PINCH_CLOSED if grasp else PINCH_OPEN
    return hand_pos

def keep_head_on_table(motor_pos):
    """Counter-rotate the neck against the waist so the head and camera keeps looking at the table."""
    pos = np.array(motor_pos, dtype=float)
    pos[3] = -motor_pos[2]
    pos[4] = HOME_POS[4] - motor_pos[1]
    return pos

def read_sample(pos_to_send, hand_pos):
    """One training sample: camera image + robot state (obs) and the command just sent (action)."""
    img_msg = cam_sub.Read(timeout=0.002)
    cur_state = cur_pos_sub.Read(timeout=0.002)
    if img_msg is None or cur_state is None:
        return None

    img = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)
    cur_pos = np.array([cur_state.motor_state[i].q for i in range(NUM_MOTOR)])
    cur_vel = np.array([cur_state.motor_state[i].dq for i in range(NUM_MOTOR)])
    grasp = int(hand_pos[HAND_SLICE["L"]][4] == PINCH_CLOSED[4] or
                hand_pos[HAND_SLICE["R"]][4] == PINCH_CLOSED[4])
    return img, cur_pos, cur_vel, np.array(pos_to_send, dtype=float), grasp, time.perf_counter()

def smooth_phase(s):
    """Minimum-jerk profile 0 -> 1: zero velocity and acceleration at start and end (no jerky starts/stops)."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def blend_rotation(start_rot, goal_rot, phase):
    """Rotation part-way (phase 0..1) between two wrist orientations (quaternion slerp)."""
    start_quat, goal_quat = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(start_quat, np.asarray(start_rot).flatten())
    mujoco.mju_mat2Quat(goal_quat, np.asarray(goal_rot).flatten())
    if np.dot(start_quat, goal_quat) < 0:
        goal_quat = -goal_quat
    angle = np.arccos(np.clip(np.dot(start_quat, goal_quat), -1.0, 1.0))
    if angle < 1e-6:
        quat = goal_quat
    else:
        quat = (np.sin((1 - phase) * angle) * start_quat + np.sin(phase * angle) * goal_quat) / np.sin(angle)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat / np.linalg.norm(quat))
    return rot.reshape(3, 3)


def current_pinch(arm):
    """Current pinch point position and wrist rotation in the IK model."""
    ik = IK_SOLVERS[arm]
    wrist = ik.configuration.get_transform_frame_to_world(ARMS[arm]["wrist"], "body")
    wrist_rot = wrist.rotation().as_matrix()
    return wrist.translation() + wrist_rot @ ik.ee_offset, wrist_rot


def joint_move(start_pos, goal_pos, duration, grasp, arm, record=False):
    """Smooth (minimum-jerk) ramp in joint space. Returns (last command, recorded samples)."""
    samples = []
    hand_pos = get_hand_pos(grasp, arm)
    t0 = time.perf_counter()
    next_tick = t0
    tick = 0

    while (t := time.perf_counter() - t0) < duration:
        phase = smooth_phase(t / duration)
        pos_to_send = (1 - phase) * start_pos + phase * goal_pos
        send_cmd(pos_to_send, hand_pos)

        if record and tick % RECORD_EVERY == 0:
            sample = read_sample(pos_to_send, hand_pos)
            if sample is not None:
                samples.append(sample)

        tick += 1
        next_tick = wait_for_next_tick(next_tick)

    return np.array(goal_pos, dtype=float), samples


def hold(cur_cmd, duration, grasp, arm):
    """Keep the arm still while the hand opens/closes (recorded)."""
    return joint_move(cur_cmd, cur_cmd, duration, grasp, arm, record=True)


def ik_move(target_point, grasp, arm, timeout=5.0, tol=1e-3):
    """Move the pinch point to target_point with IK_Solver (recorded)."""
    ik = IK_SOLVERS[arm]
    samples = []
    hand_pos = get_hand_pos(grasp, arm)
    target_point = np.asarray(target_point, dtype=float)
    start_point, start_rot = current_pinch(arm)
    duration = max(1.0, np.linalg.norm(target_point - start_point) / IK_SPEED)
    t0 = time.perf_counter()
    next_tick = t0
    last_time = t0
    tick = 0
    pos_to_send = ik.configuration.q[ik.motor_qpos].copy()

    while True:
        now = time.perf_counter()
        t = now - t0
        if t >= duration and (ik.reached(tol) or t >= duration + timeout):
            break

        phase = smooth_phase(t / duration)
        ik.ee_rot = blend_rotation(start_rot, ARMS[arm]["fingers_down"], phase)
        ik.set_target(start_point + phase * (target_point - start_point))

        dt = min(now - last_time, 0.01) if tick > 0 else CONTROL_DT     # real elapsed time per tick
        last_time = now
        pos_to_send = keep_head_on_table(ik.step(dt))
        send_cmd(pos_to_send, hand_pos)

        if tick % RECORD_EVERY == 0:
            sample = read_sample(pos_to_send, hand_pos)
            if sample is not None:
                samples.append(sample)

        tick += 1
        next_tick = wait_for_next_tick(next_tick)

    if not ik.reached(tol):
        err = np.linalg.norm(ik.hand_task.compute_error(ik.configuration)[:3]) * 1000
        print(f"  [warn] IK did not reach {np.round(target_point, 3)} ({err:.1f} mm off)")
    return pos_to_send, samples

def record_episode(ep, nut_pos, target_pos):
    """Scripted pick-and-place with IK_Solver; saves images, observations, actions and the prompt."""
    arm = "L" if nut_pos[1] > 0 else "R"
    route, ik = ROUTES[arm], IK_SOLVERS[arm]

    up = np.array([0.0, 0.0, 1.0])
    nut = np.array([nut_pos[0] + NUT_CENTRE_OFFSET_X, nut_pos[1], NUT_Z])
    target = np.array([target_pos[0], target_pos[1], NUT_Z])

    # Recorded demo
    samples = []
    pos_to_send = HOME_POS.copy()
    pos_to_send, s = hold(pos_to_send, 0.5, grasp=False, arm=arm)
    samples += s
    for goal_pos in route["raise_arm"]:
        pos_to_send, s = joint_move(pos_to_send, goal_pos, 2.0, grasp=False, arm=arm, record=True)
        samples += s
    ik.sync(pos_to_send)
    pos_to_send, s = hold(pos_to_send, 0.5, grasp=False, arm=arm)
    samples += s   # 0.5 s of still frames before moving
    pos_to_send, s = ik_move(nut + APPROACH_HEIGHT * up, grasp=False, arm=arm, tol=TRAVEL_TOL)
    samples += s   # above nut
    pos_to_send, s = ik_move(nut + GRASP_HEIGHT * up, grasp=False, arm=arm)
    samples += s   # descend
    pos_to_send, s = hold(pos_to_send, 1.5, grasp=True, arm=arm)
    samples += s   # close hand
    pos_to_send, s = ik_move(nut + LIFT_HEIGHT * up, grasp=True, arm=arm, tol=TRAVEL_TOL)
    samples += s   # lift
    pos_to_send, s = ik_move(target + LIFT_HEIGHT * up, grasp=True, arm=arm, tol=TRAVEL_TOL)
    samples += s   # carry
    pos_to_send, s = ik_move(target + PLACE_HEIGHT * up, grasp=True, arm=arm, timeout=1.5)
    samples += s   # lower: open hand at most 1.5 s after reaching the target block
    pos_to_send, s = hold(pos_to_send, 0.6, grasp=False, arm=arm)
    samples += s   # open hand
    pos_to_send, s = ik_move(target + LIFT_HEIGHT * up, grasp=False, arm=arm, tol=TRAVEL_TOL)
    samples += s   # retreat
    pos_to_send, s = joint_move(pos_to_send, route["above_table"], 3.0, grasp=False, arm=arm, record=True)
    samples += s
    for goal_pos in route["lower_arm"]:
        pos_to_send, s = joint_move(pos_to_send, goal_pos, 2.0, grasp=False, arm=arm, record=True)
        samples += s

    return samples, nut, target

def make_episode(ep, samples, nut_pos, target_pos):
    return dict(
        obs_img=[s[0] for s in samples],
        obs_pos=np.array([s[1] for s in samples]),
        obs_vel=np.array([s[2] for s in samples]),
        cmd_pos=np.array([s[3] for s in samples]),
        cmd_grasp=np.array([s[4] for s in samples], dtype=np.uint8),
        obs_time=np.array([s[5] for s in samples]),
        prompt=TASK_PROMPT, nut_pos=nut_pos, target_pos=target_pos, episode_id=ep
    )

def save_dataset(data_to_save):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "demonstrations.pkl"
    with open(path, "wb") as f:
        pickle.dump(data_to_save, f)
    print("Saved demonstrations")

if __name__ == "__main__":
    demos_saved_ep, attempt = 0, 0
    N_EPISODES = args.episodes
    STARTUP_WAIT = 2.5  # let scene.py load and the sim settle before the first episode
    episodes_to_save = []
    try:
        go_home()
        # print(f"Waiting {STARTUP_WAIT:.0f} s for the scene to settle")
        hold_home(STARTUP_WAIT)
        nut_pos, target_pos, _ = reset_sim(demos_saved_ep)
        while demos_saved_ep < N_EPISODES:
            hold_home(2.0)
            samples, nut, target = record_episode(demos_saved_ep, nut_pos, target_pos)
            go_home()
            nut_pos, target_pos, _bool = reset_sim(demos_saved_ep+1)
            attempt+=1
            if _bool:
                episodes_to_save.append(make_episode(demos_saved_ep, samples, nut, target))
                demos_saved_ep+=1
                print(f"collected episode number {demos_saved_ep}")
        print(f"Done: {demos_saved_ep} episodes saved in {attempt} attempts")
        go_home()
    except KeyboardInterrupt:
        go_home()
    finally:
        save_dataset(episodes_to_save) 
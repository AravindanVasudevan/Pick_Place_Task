import numpy as np
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DEFAULT_CKPT_DIR = ROOT / "task_imitation_learning" / "checkpoints" / "best"
SCENE_XML = ROOT / "task_imitation_learning" / "scene" / "pick_place_scene.xml"

import argparse
parser = argparse.ArgumentParser(description="Evaluation for the pick place environment")
parser.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CKPT_DIR), help="checkpoint folder")
parser.add_argument("--episodes", type=int, default=10, help="number of evaluation episodes")
args = parser.parse_args()

import torch
from model.model import IL_Model, IMAGE_KEY, STATE_KEY
from pndbotics_sdk_py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from pndbotics_sdk_py.idl.default import pnd_adam_msg_dds__LowCmd_, pnd_adam_msg_dds__HandCmd_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowCmd_, LowState_, HandCmd_
from msg.camera_msg import CameraImage
from msg.reset_msg import SimResetSignal

from utils.ik_solver import IK_Solver
from utils.timing import CONTROL_DT, RECORD_EVERY, wait_for_next_tick
from utils.utils import decode_image

EPISODE_TIME = 45.0 # Based on runs for data collections

NUM_MOTOR = 19
OBS_DIM = NUM_MOTOR + 1
ACTION_DIM = NUM_MOTOR + 1
IMG_H, IMG_W = 480, 640
CHUNK_SIZE = 50

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

PINCH_OPEN = [0, 0, 0, 400, 600, 1000]
PINCH_CLOSED = [0, 0, 0, 400, 300, 1000]
HAND_HOME_POS = np.array(PINCH_OPEN + PINCH_OPEN, dtype=int)
HAND_SLICE = {"R": slice(6, 12), "L": slice(0, 6)}

# Only used for the gravity torque in send_cmd
IK_SOLVERS = {"R": IK_Solver(SCENE_XML,
                             active_joints=[],
                             num_motors=NUM_MOTOR,
                             ee_body="wristRollRight",
                             ee_offset=[0.0, 0.0, 0.0],
                             base_body="lifting_Columns")}

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

def active_arm(motor_pos):
    """The arm the policy is moving: the one furthest from HOME (Left = motors 5-11, Right = 12-18)."""
    left = np.abs(motor_pos[5:12] - HOME_POS[5:12]).sum()
    right = np.abs(motor_pos[12:19] - HOME_POS[12:19]).sum()
    return "L" if left > right else "R"

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
    """Ask scene.py to check the last episode and randomize; returns nut_pos, target_pos, success of last episode."""
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
    """Copy the simulator's base pose into the gravity model."""
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

def run_policy(model, duration):
    """Policy at 20 Hz (every RECORD_EVERY control ticks), commands interpolated at CONTROL_DT in between."""
    model.reset()
    grasp = False
    cur_cmd = HOME_POS.copy()
    start_cmd, goal_cmd = cur_cmd.copy(), cur_cmd.copy()
    t0 = time.perf_counter()
    next_tick = t0
    tick = 0

    while time.perf_counter() - t0 < duration:
        if tick % RECORD_EVERY == 0:
            img_msg = cam_sub.Read(timeout=0.05)
            state_msg = cur_pos_sub.Read(timeout=0.05)
            if img_msg is not None and state_msg is not None:
                cur_pos = np.array([state_msg.motor_state[i].q for i in range(NUM_MOTOR)])
                obs = {IMAGE_KEY: decode_image(np.frombuffer(bytes(img_msg.data), dtype=np.uint8), IMG_H, IMG_W),
                       STATE_KEY: torch.tensor(np.append(cur_pos, float(grasp)), dtype=torch.float32)}
                action = model.act(obs)[0].numpy()
                start_cmd, goal_cmd = cur_cmd.copy(), action[:NUM_MOTOR]
                grasp = bool(action[NUM_MOTOR] > 0.5)

        phase = min(1.0, (tick % RECORD_EVERY + 1) / RECORD_EVERY)
        cur_cmd = start_cmd + phase * (goal_cmd - start_cmd)
        send_cmd(cur_cmd, get_hand_pos(grasp, active_arm(cur_cmd)))
        tick += 1
        next_tick = wait_for_next_tick(next_tick)

    return cur_cmd

def run_episode(model):
    run_policy(model, EPISODE_TIME)


if __name__ == "__main__":
    STARTUP_WAIT = 2.5
    model = IL_Model.load(args.checkpoint_path)
    print(f"loaded {args.checkpoint_path}", flush=True)

    successes = 0

    try:
        go_home()
        hold_home(STARTUP_WAIT)
        _, _, _ = reset_sim(0)

        for ep in range(args.episodes):
            hold_home(2.0)
            run_episode(model)
            go_home()
            _, _, success = reset_sim(ep + 1)

            successes += int(success)
            print(f"{'success' if success else 'fail'}", flush=True)
        print(f"Success rate: {successes}/{args.episodes}", flush=True)
        go_home()
    except KeyboardInterrupt:
        go_home()
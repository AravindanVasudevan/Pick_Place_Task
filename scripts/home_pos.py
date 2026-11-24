import time
import sys
import numpy as np
import cv2

from pndbotics_sdk_py.core.channel import ChannelPublisher, ChannelSubscriber,ChannelFactoryInitialize
from pndbotics_sdk_py.idl.default import pnd_adam_msg_dds__LowCmd_
from pndbotics_sdk_py.idl.default import pnd_adam_msg_dds__LowState_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowCmd_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowState_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import HandCmd_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import HandState_
from pndbotics_sdk_py.idl.default import pnd_adam_msg_dds__HandCmd_

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Add project paths
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_DIR))

# custom camera msg import
from task_imitation_learning.msg.camera_msg import CameraImage

ADAM_U_NUM_MOTOR = 19
KP_CONFIG = [
    405.0,  # waistRoll (0)
    405.0,  # waistPitch (1)
    205.0,  # waistYaw (2)
    9.0,   # neckYaw (3)
    9.0,   # neckPitch (4)
    180.0,  # shoulderPitch_Left (5)
    180.0,   # shoulderRoll_Left (6)
    9.0,   # shoulderYaw_Left (7)
    9.0,   # elbow_Left (8)
    9.0,   # wristYaw_Left (9)
    9.0,   # wristPitch_Left (10)
    9.0,   # wristRoll_Left (11)
    180.0,  # shoulderPitch_Right (12)
    180.0,   # shoulderRoll_Right (13)
    9.0,   # shoulderYaw_Right (14)
    9.0,   # elbow_Right (15)
    9.0,   # wristYaw_Right (16)
    9.0,   # wristPitch_Right (17)
    9.0    # wristRoll_Right (18)
]

KD_CONFIG = [
    6.75,   # waistRoll (0)
    6.75,   # waistPitch (1)
    3.42,   # waistYaw (2)
    0.9,   # neckYaw (3)
    0.9,   # neckPitch (4)
    1.8,   # shoulderPitch_Left (5)
    1.8,   # shoulderRoll_Left (6)
    0.9,   # shoulderYaw_Left (7)
    0.9,   # elbow_Left (8)
    0.9,   # wristYaw_Left (9)
    0.9,   # wristPitch_Left (10)
    0.9,   # wristRoll_Left (11)
    1.8,   # shoulderPitch_Right (12)
    1.8,   # shoulderRoll_Right (13)
    0.9,   # shoulderYaw_Right (14)
    0.9,   # elbow_Right (15)
    0.9,   # wristYaw_Right (16)
    0.9,   # wristPitch_Right (17)
    0.9    # wristRoll_Right (18)
]

home_pos_all = np.array([0, 0, 0,
                     0, 0.5,
                     0, 0, 0, 0,
                     0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0
], dtype=float)

hand_open_pos = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000], dtype=int)
# hand_close_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)

dt = 0.0025
runing_time = 0.0

input("Press enter to start")

if __name__ == '__main__':

    if len(sys.argv) <2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(1, sys.argv[1])

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    cmd = pnd_adam_msg_dds__LowCmd_(19)

    hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
    hand_pub.Init()
    hand_cmd = pnd_adam_msg_dds__HandCmd_()

    cam_sub = ChannelSubscriber("rt/camera/image", CameraImage)
    cam_sub.Init()


    # video_writer = None
    # frame_count = 0
    # target_frames = 150
    # video_start_time = None

    try:
        while True:
            step_start = time.perf_counter()

            runing_time += dt

            if runing_time < 3.0:
                phase = np.tanh((runing_time - 3.0) / 1.2)

                for i in range(ADAM_U_NUM_MOTOR):
                    cmd.motor_cmd[i].q = (
                        phase * home_pos_all[i]
                        + (1 - phase) * home_pos_all[i]
                    )
                    cmd.motor_cmd[i].kp = KP_CONFIG[i]
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = KD_CONFIG[i]
                    cmd.motor_cmd[i].tau = 0.0

                for i in range(12):
                    hand_cmd.position[i] = hand_open_pos[i]

            pub.Write(cmd)
            hand_pub.Write(hand_cmd)

            # if runing_time >= 3.0 and frame_count < target_frames:
            #     camera_msg = cam_sub.Read()
            #     if camera_msg is not None:
            #         jpeg_bytes = bytes(camera_msg.data)
            #         image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            #         if image is not None:
            #             if video_writer is None:
            #                 height, width = image.shape[:2]

            #                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            #                 video_writer = cv2.VideoWriter("home_position_test.mp4", fourcc, 30.0, (width, height))

            #                 video_start_time = time.perf_counter()

            #             video_writer.write(image)
            #             frame_count += 1

            #             if frame_count % 30 == 0:
            #                 print(f"Captured {frame_count}/150 frames")

            # if frame_count >= target_frames and video_writer is not None:
            #     video_writer.release()
            #     video_writer = None
            #     print("Video saved: home_position_test.mp4")
            #     print(f"Total frames captured: {frame_count}")

            time_until_next_step = dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
    except KeyboardInterrupt:
        print("\nStopping robot commands...")

        # Disable motor position and velocity control gains
        for i in range(ADAM_U_NUM_MOTOR):
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0

        # Publish the stop command several times
        for _ in range(20):
            pub.Write(cmd)
            time.sleep(dt)

        print("Stop command sent. Exiting.")
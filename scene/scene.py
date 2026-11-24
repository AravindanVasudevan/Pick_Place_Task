import argparse
import os

parser = argparse.ArgumentParser(description="Pick-and-place scene (MuJoCo + PND DDS bridge)")
parser.add_argument("--headless", action="store_true",
                    help="run without the viewer window for data collection and evaluation")
args = parser.parse_args()
if args.headless:
    os.environ.setdefault("MUJOCO_GL", "egl")


import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import cv2
import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = Path(__file__).resolve().parents[1]
SCENE_DIR = Path(__file__).resolve().parent
PND_SIM_DIR = ROOT / "pnd_mujoco" / "simulate_python"

# Add project paths
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PND_SIM_DIR))
sys.path.insert(0, str(PROJECT_DIR))

# custom camera msg import
from msg.camera_msg import CameraImage
from msg.reset_msg import SimResetSignal

# Import PND SDK
from pndbotics_sdk_py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber

# Load PND configuration
import importlib.util

spec = importlib.util.spec_from_file_location("config", PND_SIM_DIR / "config.py")

pnd_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pnd_config)

# Tell PND to use Adam-U instead of Adam-U Omnipicker
pnd_config.ROBOT = "adam_u"
pnd_config.ROBOT_SCENE = str(SCENE_DIR / "pick_place_scene.xml")

# Make the PND bridge use this configuration
sys.modules["config"] = pnd_config

# Import PND bridges AFTER configuring the robot
from pndbotics_sdk_py_bridge import pndSdkBridge
from pndbotics_ros_bridge import pndRos2Bridge


locker = threading.Lock()

def load_model_with_crown_camera(scene_xml):
    """Load the scene and attach a camera to the top of the head - neckPitch"""
    spec = mujoco.MjSpec()
    spec.from_file(scene_xml)
    cam = spec.find_body("neckPitch").add_camera()
    cam.name = "crown_camera"
    cam.pos = [0.10, -0.02, 0.15]
    cam.quat = [0.6381, 0.3047, -0.3047, -0.6381]
    cam.fovy = 60

    spec.find_body("lifting_Columns").pos = [0.0, 0.0, 1.0333]
    base_fix = spec.add_equality()
    base_fix.type = mujoco.mjtEq.mjEQ_WELD
    base_fix.name1 = "lifting_Columns"

    return spec.compile()

mj_model = load_model_with_crown_camera(pnd_config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)
camera_data = mujoco.MjData(mj_model)   # Camera's copy of simulation data
azimuth = mj_model.vis.global_.azimuth
elevation = mj_model.vis.global_.elevation


# Domain randomaization for target block and nut pos
TABLE_CENTER_XY = np.array([0.5, 0.0])
XY_Range = 0.05
MIN_SEPARATION = 0.10
NUT_X_RANGE = (0.425, 0.50)    # nut spawn area X
NUT_Y_RANGE = (-0.075, 0.075)   # nut spawn area Y

NUT_INIT = mj_data.joint("nut_freejoint").qpos.copy()
TARGET_MOCAP = mj_model.body("target_block").mocapid[0]
TARGET_INIT = mj_model.body("target_block").pos.copy()

def randomize_on_reset():
    for _ in range(100):
        nut_xy = np.array([np.random.uniform(*NUT_X_RANGE), np.random.uniform(*NUT_Y_RANGE)])
        target_xy = TABLE_CENTER_XY + np.random.uniform(-XY_Range, XY_Range, 2)
        if np.linalg.norm(nut_xy - target_xy) >= MIN_SEPARATION:
            break

    nut = mj_data.joint("nut_freejoint")
    nut.qpos[:] = NUT_INIT
    nut.qpos[:2] = nut_xy
    nut.qvel[:] = 0.0

    mj_data.mocap_pos[TARGET_MOCAP] = TARGET_INIT
    mj_data.mocap_pos[TARGET_MOCAP][:2] = target_xy

def reset_handler(done_pub):
    def on_reset(msg: SimResetSignal):
        with locker:
            nut_xy = mj_data.joint("nut_freejoint").qpos[:2] + [0.008, 0]
            half = mj_model.geom("target_visual").size[:2]
            success = bool(np.all(np.abs(nut_xy - mj_data.mocap_pos[TARGET_MOCAP][:2]) <= half))

            randomize_on_reset()
            mujoco.mj_forward(mj_model, mj_data)
            nut_pos = mj_data.joint("nut_freejoint").qpos[:3].tolist()
            target_pos = mj_data.mocap_pos[TARGET_MOCAP].tolist()
            base_pos = mj_data.qpos[0:3].tolist()
            base_quat = mj_data.qpos[3:7].tolist()
        done_pub.Write(SimResetSignal(episode_id=msg.episode_id, nut_pos=nut_pos, target_pos=target_pos,
                                      base_pos=base_pos, base_quat=base_quat, success=success))
    return on_reset



class HeadlessViewer:
    """Stands in for the MuJoCo viewer when running headless (same is_running / sync / close calls)."""
    def __init__(self):
        self.running = True
    def is_running(self):
        return self.running
    def sync(self):
        pass
    def close(self):
        self.running = False

if args.headless:
    viewer = HeadlessViewer()
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    viewer.cam.azimuth = azimuth
    viewer.cam.elevation = elevation
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0, 0, 1]

mj_model.opt.timestep = pnd_config.SIMULATE_DT

def update_camera_data():
    with locker:
        camera_data.qpos[:] = mj_data.qpos
        camera_data.qvel[:] = mj_data.qvel

        camera_data.mocap_pos[:] = mj_data.mocap_pos
        camera_data.mocap_quat[:] = mj_data.mocap_quat

        camera_data.time = mj_data.time

    mujoco.mj_forward(mj_model, camera_data)


CAMERA_FPS = 30
CAMERA_PERIOD = 1.0 / CAMERA_FPS

def publish_camera_image(renderer, camera_pub, frame_id):

    renderer.update_scene(camera_data, camera="crown_camera")

    rgb = renderer.render()

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    success, encoded = cv2.imencode(".jpg", bgr)

    if not success:
        return

    msg = CameraImage(
        frame_id=frame_id,
        timestamp=time.time_ns(),
        width=640,
        height=480,
        encoding="jpeg",
        data=encoded.reshape(-1).tolist()
    )

    camera_pub.Write(msg)

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    if pnd_config.SDK_TYPE == "ROS2":
        ChannelFactoryInitialize(pnd_config.DOMAIN_ID)
        pnd = pndRos2Bridge(mj_model, mj_data)
    elif pnd_config.SDK_TYPE == "DDS":
        ChannelFactoryInitialize(pnd_config.DOMAIN_ID, pnd_config.INTERFACE)
        pnd = pndSdkBridge(mj_model, mj_data)

    camera_pub = ChannelPublisher("rt/camera/image", CameraImage)
    camera_pub.Init()

    reset_done_pub = ChannelPublisher("rt/sim/reset_done", SimResetSignal)
    reset_done_pub.Init()
    reset_request_sub = ChannelSubscriber("rt/sim/reset_request", SimResetSignal)
    reset_request_sub.Init(reset_handler(reset_done_pub), 10)

    camera_thread = Thread(target=CameraThread, args=(camera_pub,))
    camera_thread.start()

    if pnd_config.USE_JOYSTICK:
        pnd.SetupJoystick(device_id=0, js_type=pnd_config.JOYSTICK_TYPE)
    if pnd_config.PRINT_SCENE_INFORMATION:
        pnd.PrintSceneInformation()

    print("scene ready start collect_demos.py / evaluate.py", flush=True)

    while viewer.is_running():

        step_start = time.perf_counter()

        locker.acquire()

        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    camera_thread.join()

def CameraThread(camera_pub):
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frame_id = 0

    try:
        while viewer.is_running():

            start_time = time.perf_counter()

            update_camera_data()

            publish_camera_image(renderer, camera_pub, frame_id)

            frame_id += 1

            elapsed = time.perf_counter() - start_time
            time.sleep(max(0, CAMERA_PERIOD - elapsed))

    finally:
        renderer.close()

def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(pnd_config.VIEWER_DT)

if __name__ == "__main__":
    try:
        sim_thread = Thread(target=SimulationThread)
        sim_thread.start()
        if not args.headless:
            viewer_thread = Thread(target=PhysicsViewerThread)
            viewer_thread.start()
            viewer_thread.join()
        sim_thread.join()
    finally:
        viewer.close()
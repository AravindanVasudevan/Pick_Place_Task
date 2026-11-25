# Imitation Learning for Pick and Place on ADAM-U in MuJoCo

## 1. Task

The ADAM-U humanoid (from [pnd_mujoco](https://github.com/pndbotics/pnd_mujoco)) picks up a hex nut (22 mm across flats, 16 mm thick) and places it on a green target on the table. The nut and target positions are randomised every episode.

The pipeline has four parts:

1. **Simulation**: MuJoCo 3.2.0, talking to the robot over DDS through the PND SDK.
2. **Demonstrations**: a scripted IK expert records about 100 successful episodes.
3. **Training**: an ACT policy with an ImageNet-pretrained ResNet18 backbone (LeRobot).
4. **Evaluation**: the trained policy controls the robot over DDS. It decides which arm to use and when to close the hand from the camera image alone.


## 3. Installation


**3.1 Project folder and virtual environment**
```bash
mkdir project && cd project
python3.10 -m venv venv_mujoco
source venv_mujoco/bin/activate
```

**3.2 This repository** 
```bash
git clone https://github.com/AravindanVasudevan/Pick_Place_Task.git
```

**3.3 PND MuJoCo and SDK** (ADAM-U model and DDS communication)
```bash
git clone https://github.com/pndbotics/pnd_mujoco.git
git clone https://github.com/pndbotics/pnd_sdk_python.git
pip install -e ./pnd_sdk_python
```

**3.4 Python dependencies**
```bash
pip install -r task_imitation_learning/requirements.txt
pip install --force-reinstall --no-deps opencv-python==5.1.0.95
```
The second command is required: LeRobot installs `opencv-python-headless`, which replaces the `cv2` used by this project.

**3.5 Nut meshes**

Copy the provided `nut.stl` and `nut (bigger rod).obj` into:
```text
task_imitation_learning/assets/object_nut/
```

## 4. Usage

Run all commands from `project/` with the virtual environment active.

> Run only **one** `scene.py` at a time.

### 4.1 Collect demonstrations

Terminal 1, the simulation (`--headless` runs without the viewer window):
```bash
python task_imitation_learning/scene/scene.py --headless
```

Terminal 2, the scripted expert:
```bash
python task_imitation_learning/scripts/collect_demos.py --episodes 100
```

Only successful episodes are saved, to `data/demonstrations.pkl`. Re-running the command overwrites this file. When it finishes, stop `scene.py` with Ctrl+C.

### 4.2 Train
```bash
python task_imitation_learning/scripts/train.py --steps 20000
```
- The last 5 episodes are used for validation.
- Every 2000 steps, the validation loss is computed and the model is saved to `checkpoints/last`; it's also saved to `checkpoints/best` when the validation loss improves.
- Optional arguments: `--batch` (default 8) and `--out` (checkpoint folder, default `checkpoints/`).

To continue training from `checkpoints/last`, pass the **total** number of steps, not the number of extra steps:
```bash
python task_imitation_learning/scripts/train.py --steps 40000 --continue_training
```

### 4.3 Evaluate

Terminal 1:
```bash
python task_imitation_learning/scene/scene.py --headless
```

Terminal 2:
```bash
python task_imitation_learning/scripts/evaluate.py --episodes 10 --checkpoint_path task_imitation_learning/checkpoints/best
```


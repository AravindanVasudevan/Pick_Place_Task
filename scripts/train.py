import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.model import IL_Model, IMAGE_KEY, STATE_KEY, ACTION_KEY

DATA_FILE = ROOT / "task_imitation_learning" / "data" / "demonstrations.pkl"
CKPT_DIR = ROOT / "task_imitation_learning" / "checkpoints"

import argparse
parser = argparse.ArgumentParser(description="Train ACT (pretrained ResNet18) on the pick and place demos")
parser.add_argument("--steps", type=int, default=20000, help="number of training steps")
parser.add_argument("--batch", type=int, default=8, help="batch size")
parser.add_argument("--out", type=str, default=str(CKPT_DIR), help="checkpoint folder")
parser.add_argument("--continue_training", action="store_true", help="continue training from last saved checkpoint")
args = parser.parse_args()



import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


NUM_MOTOR = 19
OBS_DIM = NUM_MOTOR + 1
ACTION_DIM = NUM_MOTOR + 1
IMG_H, IMG_W = 480, 640
CHUNK_SIZE = 50
N_VAL_EPISODES = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def decode_image(jpeg_bytes):
    bgr = cv2.imdecode(np.asarray(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def episode_arrays(ep):
    """Inputs (state) and outputs (actions) of one episode as arrays."""
    grasp = np.asarray(ep["cmd_grasp"], dtype=np.float32)
    hand_now = np.concatenate([[0.0], grasp[:-1]])
    states = np.concatenate([np.asarray(ep["obs_pos"]), hand_now[:, None]], axis=1)
    actions = np.concatenate([np.asarray(ep["cmd_pos"]), grasp[:, None]], axis=1)
    return states.astype(np.float32), actions.astype(np.float32)


class DemoDataset(Dataset):
    """One item = image + state at step t, and the next CHUNK_SIZE actions."""

    def __init__(self, episodes):
        self.images = [ep["obs_img"] for ep in episodes]
        self.arrays = [episode_arrays(ep) for ep in episodes]
        self.index = [(e, t) for e, (states, _) in enumerate(self.arrays) for t in range(len(states))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        states, actions = self.arrays[e]

        chunk = actions[t:t + CHUNK_SIZE]
        is_pad = np.zeros(CHUNK_SIZE, dtype=bool)
        if len(chunk) < CHUNK_SIZE:
            is_pad[len(chunk):] = True
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], CHUNK_SIZE - len(chunk), axis=0)])

        return {
            IMAGE_KEY: decode_image(self.images[e][t]),
            STATE_KEY: torch.from_numpy(states[t]),
            ACTION_KEY: torch.from_numpy(chunk),
            "action_is_pad": torch.from_numpy(is_pad),
        }


def compute_stats(episodes):
    """Mean / std to normalise state and action; ImageNet mean / std for the pretrained ResNet18."""
    states, actions = zip(*[episode_arrays(ep) for ep in episodes])
    states, actions = np.concatenate(states), np.concatenate(actions)
    to_t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
    return {
        IMAGE_KEY: {"mean": to_t(IMAGENET_MEAN).view(3, 1, 1), "std": to_t(IMAGENET_STD).view(3, 1, 1)},
        STATE_KEY: {"mean": to_t(states.mean(0)), "std": to_t(states.std(0) + 1e-6)},
        ACTION_KEY: {"mean": to_t(actions.mean(0)), "std": to_t(actions.std(0) + 1e-6)},
    }


def val_loss(model, loader):
    losses = []
    with torch.no_grad():
        for batch in loader:
            loss, _ = model(batch)
            losses.append(loss.item())
    return float(np.mean(losses))


def main():
    out_dir = Path(args.out)

    print("loading", DATA_FILE, flush=True)
    with open(DATA_FILE, "rb") as f:
        episodes = pickle.load(f)
    train_eps, val_eps = episodes[:-N_VAL_EPISODES], episodes[-N_VAL_EPISODES:]
    print(f"{len(train_eps)} train / {len(val_eps)} val episodes", flush=True)

    if args.continue_training:
        model = IL_Model.load(out_dir / "last")
    else:
        model = IL_Model(ACTION_DIM, OBS_DIM, CHUNK_SIZE, (IMG_H, IMG_W))
        model.set_stats(compute_stats(train_eps))
    model.train()

    train_loader = DataLoader(DemoDataset(train_eps), batch_size=args.batch, shuffle=True,
                              num_workers=4, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(DemoDataset(val_eps), batch_size=args.batch, num_workers=2)

    cfg = model.config
    optimizer = torch.optim.AdamW(model.policy.get_optim_params(), lr=cfg.optimizer_lr,
                                  weight_decay=cfg.optimizer_weight_decay)

    step, best_val = 0, float("inf")
    if args.continue_training:
        state = torch.load(out_dir / "last" / "train_state.pt")
        optimizer.load_state_dict(state["optimizer"])
        step, best_val = state["step"], state["best_val"]

    while step < args.steps:
        for batch in train_loader:
            loss, info = model(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            step += 1

            if step % 200 == 0:
                print(f"step {step}  loss {loss.item():.4f}  l1 {info['l1_loss']:.4f}  kld {info['kld_loss']:.4f}", flush=True)

            if step % 2000 == 0 or step == args.steps:
                val = val_loss(model, val_loader)
                print(f"step {step}  val loss {val:.4f}", flush=True)
                if val < best_val:
                    best_val = val
                    model.save(out_dir / "best")
                model.save(out_dir / "last")
                torch.save({"optimizer": optimizer.state_dict(), "step": step, "best_val": best_val}, out_dir / "last" / "train_state.pt")

            if step >= args.steps:
                break

    print(f"done, best val loss {best_val:.4f}, saved in {out_dir}", flush=True)


if __name__ == "__main__":
    main()

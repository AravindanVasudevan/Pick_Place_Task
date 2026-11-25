import numpy as np
import cv2
import torch


def decode_image(jpeg_bytes, IMG_H, IMG_W):
    bgr = cv2.imdecode(np.asarray(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
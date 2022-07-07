"""
Continuously capture images from a webcam and write to a Redis store.
Usage:
   python recorder.py [width] [height]
"""

import itertools
import sys

import redis
import imageio
import torch

from animate import normalize_kp
from demo import load_checkpoints
import numpy as np

from skimage.transform import resize

import os

import time


import imageio
import torch
from tqdm import tqdm
from animate import normalize_kp
from demo import load_checkpoints
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import img_as_ubyte
from skimage.transform import resize
import cv2
import os

import time

import cv2


# Retrieve command line arguments.
WIDTH = None if len(sys.argv) <= 1 else int(sys.argv[1])
HEIGHT = None if len(sys.argv) <= 2 else int(sys.argv[2])

print("[INFO] loading source image and checkpoint...")
checkpoint_path = 'vox-cpk.pth.tar'  # 얘는 고정. 학습된 모델
source_path = 'Inputs/orlando_bloom.jpg'
source_image = imageio.imread(source_path)
source_image = resize(source_image,(256,256))[..., :3]
# 모델 다운
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                                          checkpoint_path=checkpoint_path)
img = cv2.imread("Inputs/orlando_bloom.jpg")
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

relative = True
adapt_movement_scale = True
cpu = False
video_path = None

cap = cv2.VideoCapture(1)
print("[INFO] Initializing front camera...")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out1 = cv2.VideoWriter('output/test.avi', fourcc, 12, (256*3 , 256), True)

cv2_source = cv2.cvtColor(source_image.astype('float32'),cv2.COLOR_BGR2RGB)
# Create video capture object, retrying until successful.
MAX_SLEEP = 5.0
CUR_SLEEP = 0.1
while True:
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        break
    print(f'not opened, sleeping {CUR_SLEEP}s')
    time.sleep(CUR_SLEEP)
    if CUR_SLEEP < MAX_SLEEP:
        CUR_SLEEP *= 2
        CUR_SLEEP = min(CUR_SLEEP, MAX_SLEEP)
        continue
    CUR_SLEEP = 0.1

# Create client to the Redis store.
store = redis.Redis()

# Set video dimensions, if given.
if WIDTH:
    cap.set(3, WIDTH)
if HEIGHT:
    cap.set(4, HEIGHT)

# Repeatedly capture current image, encode it, convert it to bytes and push
# it to Redis database. Then create unique ID, and push it to database as well.
with torch.no_grad() :
    predictions = []
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not cpu:
        source = source.cuda()
    kp_source = kp_detector(source)
    count = 0
    for count in itertools.count(1):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # _, image = cap.read()
        if frame is None:
            time.sleep(0.5)
            continue
        if ret == True:

            if not video_path:
                x = 143
                y = 87
                w = 322
                h = 322
                frame = frame[y:y + h, x:x + w]
            frame1 = resize(frame, (256, 256))[..., :3]

            if count == 1:
                source_image1 = frame1
                source1 = torch.tensor(source_image1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                kp_driving_initial = kp_detector(source1)

            frame_test = torch.tensor(frame1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

            driving_frame = frame_test
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source,
                                   kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial,
                                   use_relative_movement=relative,
                                   use_relative_jacobian=relative,
                                   adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            im = np.transpose(out['prediction'].data.cpu().mul(255).numpy().astype(np.uint8), [0, 2, 3, 1])[0]
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            joinedFrame = np.concatenate((cv2_source, im, frame1), axis=1)
            count += 1
            img = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            store.set('image', np.array(frame).tobytes())
            store.set('image_id', os.urandom(4))
            print(count)
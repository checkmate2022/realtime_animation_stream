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
import argparse

from pyimagesearch.motion_detection.singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response, make_response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2


import cv2 
from flask import Flask, render_template, Response




app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


# gen함수가 작동하면 무한 while문을 통해 카메라를 읽고 애니메이션화가 이루어져야함
def gen():
    """Video streaming generator function."""
    global vs, outputFrame, lock, source_image, kp_detector, generator

    checkpoint_path = 'vox-cpk.pth.tar'  # 얘는 고정. 학습된 모델
    source_path = 'Inputs/orlando_bloom.jpg'
    # if args['input_video']:  # 비디오.mp4를 input으로 넣었을 때
    #    video_path = args['input_video']
    # else:  # 여기가 웹캠인듯
    source_image = imageio.imread(source_path)  # input 이미지
    source_image = resize(source_image, (256, 256))[..., :3]  # input 이미지 사이즈 재설정

    # 모델 다운
    generator, kp_detector = load_checkpoints(config_path='../ImageAnimation-new/config/vox-256.yaml',
                                              checkpoint_path=checkpoint_path)

    img = cv2.imread("Inputs/orlando_bloom.jpg")
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    #frame2 = cv2.imencode('.jpg', img)[1].tobytes()    # 지금 frame2을 출력하고 있는 것
    #yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

    relative = True
    adapt_movement_scale = True
    cpu = True
    video_path = None


    if video_path:
        cap = cv2.VideoCapture(video_path)
        print("[INFO] Loading video from the given path")
    else:
        cap = cv2.VideoCapture(0)
        print("[INFO] Initializing front camera...")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out1 = cv2.VideoWriter('output/test.avi', fourcc, 12, (256 * 3, 256), True)

    cv2_source = cv2.cvtColor(source_image.astype('float32'), cv2.COLOR_BGR2RGB)      ##########
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)  # 모델에 들어감
        # source가 모델에 들어가기 위해 tensor로 바뀌는데, 데이터 타입이 0~1사이인 것을 알 수 있음

        #if not cpu:
        #    source = source.cuda()
        kp_source = kp_detector(source)
        count = 0
        while (True):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if ret == True:

                if not video_path:
                    x = 143
                    y = 87
                    w = 322
                    h = 322
                    frame = frame[y:y + h, x:x + w]
                frame1 = resize(frame, (256, 256))[..., :3]            # frame1은 사용자 모습의 첫 프레임

                if count == 0:
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
                #im = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                im = np.transpose(out['prediction'].data.cpu().mul(255).numpy().astype(np.uint8), [0, 2, 3, 1])[0]    # 여기 수정
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                joinedFrame = np.concatenate((cv2_source, im, frame1), axis=1)

                #cv2.imshow('Test', im)   # im 나옴
                #out1.write(img_as_ubyte(joinedFrame))
                count += 1

                #img = cv2.imread("Inputs/orlando_bloom.jpg")
                img = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
               # re, buf = cv2.imencode('.jpg', img)[1].tobytes()

                #ret, buffer = cv2.imencode('.jpg', img)
                #frame = buffer.tobytes()

                # cv2.imshow('Test', im)     # 잘 나옴
                #cv2.imshow('Test', buffer)

                #frame = cv2.imencode('.jpg', img)[1].tobytes()

                #yield (b'--frame\r\n'
                #       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                frame = cv2.imencode('.jpg', img)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                # time.sleep(0.3)


                #cv2.imshow('Test2', frame2)   # 얘가 안나옴



                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            else:
                break

        #cap.release()
        #out1.release()
        #cv2.destroyAllWindows()




@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    


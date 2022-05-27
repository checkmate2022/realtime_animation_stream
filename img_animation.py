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
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
#vs = VideoStream(src=0).start()
#time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


# 이 함수에서 먼저 모델을 담아놓고
def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock, source_image, kp_detector, generator
    # initialize the motion detector and the total number of frames
    # read thus far
    source_path = 'Inputs/orlando_bloom.jpg'  # 이후에 여기에 아바타 이미지를 넣어주면 됨
    checkpoint_path = 'vox-cpk.pth.tar'  # 얘는 고정. 학습된 모델

    # if args['input_video']:  # 비디오.mp4를 input으로 넣었을 때
    #    video_path = args['input_video']
    # else:  # 여기가 웹캠인듯
    source_image = imageio.imread(source_path)    # input 이미지
    source_image = resize(source_image, (256, 256))[..., :3]  # input 이미지 사이즈 재설정

    generator, kp_detector = load_checkpoints(config_path='../ImageAnimation-new/config/vox-256.yaml', checkpoint_path=checkpoint_path)    # 여기서 모델 다운받아서 저장하는듯


# 웹페이지에 while문을 통해 실시간으로 출력하는 함수
def gen():
    # grab global references to the output frame and lock variables
    global outputFrame, lock, source_image, kp_detector, generator

    # output 폴더 만들어서
    if not os.path.exists('output'):
        os.mkdir('output')

    relative = True
    adapt_movement_scale = True
    cpu = True

    cap = cv2.VideoCapture(0)
    print("[INFO] Initializing front camera...")

    # out1은 사용자가 돌려본 결과물이 저장되나봄
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out1 = cv2.VideoWriter('output/result.avi', fourcc, 12, (256 * 3, 256), True)

    cv2_source = cv2.cvtColor(source_image.astype('float32'), cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        #if not cpu:  # gpu쓸 때
        #    source = source.cuda()
        kp_source = kp_detector(source)
        count = 0

        # acquire the lock, set the output frame, and release the
        # lock
        # with lock:
            # outputFrame = frame.copy()

    video_path=None
    count=0

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            #if outputFrame is None:
            #    continue
            ret, frame = cap.read()  # 카메라에 들어오는 걸 프레임별로 읽음
            frame = cv2.flip(frame, 1)
            if ret == True:     # ret이 True면 프레임을 제대로 읽었다는 뜻

                if not video_path:
                    x = 143
                    y = 87
                    w = 322
                    h = 322
                    frame = frame[y:y + h, x:x + w]
                frame1 = resize(frame, (256, 256))[..., :3]

                if count == 0:
                    source_image1 = frame1
                    source1 = torch.tensor(source_image1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                    kp_driving_initial = kp_detector(source1)

                frame_test = torch.tensor(frame1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

                driving_frame = frame_test

                #if not cpu:    # 이건 gpu쓸 때 주석 열어줘야 됨
                #    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source,
                                       kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial,
                                       use_relative_movement=relative,
                                       use_relative_jacobian=relative,
                                       adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)      # 여기서 애니메이션화되는 듯
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                im = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                joinedFrame = np.concatenate((cv2_source, im, frame1), axis=1)  # Numpy 배열들을 하나로 합치는데 이용
                # 데모에 보이는 세 개 화면이 joinedFrame에 해당. 우리는 im만 필요

                # joinedFrame = np.concatenate((cv2_source, im), axis=1)
                # joinedFrame-> cv2_source(input이미지 자체), im(애니메이션 결과**), frame1(실시간 웹캠 내 모습)을 합치는것

                cv2.imshow('Result', im)  # 나오는 결과 제목이 Result 안중요
                # outputFrame = im.copy()
                # cv2.imshow('Result', im)
                # out1.write(img_as_ubyte(joinedFrame))     # 얘는 output폴더에 저장하는거고
                count += 1
                # outputFrame = im
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

            else:
                break

            # cap.release()
            # out1.release()
            # cv2.destroyAllWindows()
            outputFrame = im      # frame으로 하면 잘 출력됨
            # with lock:
            # outputFrame = frame

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)    # opencv에서 이미지는 배열형식이라 웹페에지는 다른 형식으로 바꿔줘야함
            if encodedImage is None:
                print("encodedimage is none")
            # yield the output frame in the byte format
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')

            # ensure the frame was successfully encoded
            if not flag:
                continue



@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()

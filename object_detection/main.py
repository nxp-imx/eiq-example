#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import cv2
import time, os
import argparse

from labels import label2string

MODEL_PATH = "../vela_models/ssd_mobilenet_v1_quant_vela.tflite"

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input',
    default='/dev/video0',
    help='input to be classified')
args = parser.parse_args()

vid = cv2.VideoCapture(args.input)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

msg = ""
total_fps = 0
total_time = 0

ret, frame = vid.read()
if (frame is None):
    print("Can't read frame from source file ", args.input)
    exit(0)

while ret:
    total_fps += 1
    loop_start = time.time()

    img = cv2.resize(frame, (width,height)).astype(np.uint8)
    input_data = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    invoke_start = time.time()
    interpreter.invoke()
    invoke_end = time.time()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    labels = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    number = interpreter.get_tensor(output_details[3]['index'])[0]
    for i in range(int(number)):
        if scores[i] > 0.5:
            box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
            x0 = max(2, int(box[1] * frame.shape[1]))
            y0 = max(2, int(box[0] * frame.shape[0]))
            x1 = int(box[3] * frame.shape[1])
            y1 = int(box[2] * frame.shape[0])

            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(frame, label2string[labels[i]], (x0, y0 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print("rectangle:(%d,%d),(%d,%d) label:%s" % 
                    (x0, y0, x1, y1, label2string[labels[i]]))

    loop_end = time.time()
    total_time += (loop_end - loop_start)

    fps = int(total_fps / total_time)
    invoke_time = int((invoke_end - invoke_start) * 1000)
    msg = "FPS:" + str(fps) + "  Invoke time:" + str(invoke_time) + "ms"
    cv2.putText(frame, msg, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    cv2.imshow("image", frame)

    ret, frame = vid.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

time.sleep(2)
vid.release()
cv2.destroyAllWindows()

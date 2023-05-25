#
# Copyright 2020-2023 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import cv2
import time, math
import numpy as np
import argparse

from face_detection import YoloFace
from face_recognition import Facenet
from face_database import FaceDatabase

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input',
    default='/dev/video0',
    help='input device')
parser.add_argument(
    '-d',
    '--delegate',
    default='',
    help='delegate path')
args = parser.parse_args()

detector = YoloFace("../models/yoloface_int8.tflite", args.delegate)
recognizer = Facenet("../models/facenet_512_int_quantized.tflite", args.delegate)
database = FaceDatabase()

def ischar(c):
    return (c >= 65 and c <= 90) or (c >= 87 and c <= 122) or c == 32

def get_inputs(img, msg):
    inputs = ""
    while True:
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, msg + inputs, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('img', img)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 or key == 141:
            break
        if key == 8 and len(inputs) > 0:
            inputs = "".join(list(inputs)[:-1])
        elif ischar(key):
            inputs += chr(key)
    return inputs

def print_longtext(img, text):
    textsize = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    raw_num = int((img.shape[1] - 60) / textsize[0])
    line_num = math.ceil(len(text) / raw_num)
    gap = textsize[1] + 10

    total_y = (int(textsize[1] / 2) + gap) * line_num + 15
    cv2.rectangle(img, (0, 0), (img.shape[1], total_y), (0, 0, 0), -1)
    for i in range(line_num):
        line = text[i  * raw_num : (i + 1) * raw_num]
        y = int((30 + textsize[1]) / 2) + i * gap + 15

        cv2.putText(img, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, lineType = cv2.LINE_AA)

if args.input.isdigit():
    cap_input = int(args.input)
else:
    cap_input = args.input
vid = cv2.VideoCapture(cap_input)
PADDING = 10
tips = "Press 'a' to add person, 'd' to delete person, 'p' to print database"
while True:
    embeddings = None

    ret, img = vid.read()
    if (ret == False):
        break
    boxes = detector.detect(img)
    for box in boxes:
        box[[0,2]] *= img.shape[1]
        box[[1,3]] *= img.shape[0]
        x1, y1, x2, y2 = box.astype(np.int32)
        h, w, _ = img.shape
        x1 = max(x1-PADDING, 0)
        x2 = min(x2+PADDING, w)
        y1 = max(y1-PADDING, 0)
        y2 = min(y2+PADDING, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)

        face = img[y1:y2, x1:x2]
        embeddings = recognizer.get_embeddings(face)
        name = database.find_name(embeddings)
        cv2.putText(img, name, (x1, y1 + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(img, tips, (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xFF
    if (key == ord('a')):
        msg = "ADD. Please input name:"
        name = get_inputs(img, msg)
        database.add_name(name, embeddings)
    elif (key == ord('d')):
        msg = "DEL. Please input name:"
        name = get_inputs(img, msg)
        database.del_name(name)
    elif (key == ord('p')):
        names = ",".join(database.get_names())
        print_longtext(img, names + "   Press any key to continue.")
        cv2.imshow('img', img)
        while cv2.waitKey(100) & 0xFF == 0xFF:
            pass

time.sleep(2)
vid.release()
cv2.destroyAllWindows()

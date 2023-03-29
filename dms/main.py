#
# Copyright 2020-2023 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import sys
import time
import argparse

from face_detection import *
from eye_landmark import EyeMesher
from face_landmark import FaceMesher
from utils import *

MODEL_PATH = pathlib.Path("../models/")
DETECT_MODEL = "face_detection_front_128_full_integer_quant.tflite"
LANDMARK_MODEL = "face_landmark_192_integer_quant.tflite"
EYE_MODEL = "iris_landmark_quant.tflite"

# turn on camera
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input',
    default='/dev/video0',
    help='input to be classified')
parser.add_argument(
    '-d',
    '--delegate',
    default='',
    help='delegate path')
args = parser.parse_args()

if args.input.isdigit():
    cap_input = int(args.input)
else:
    cap_input = args.input
cap = cv2.VideoCapture(cap_input)
ret, image = cap.read()
if not ret:
    print("Can't read frame from source file ", args.input)
    sys.exit(-1)

h, w, _ = image.shape
target_dim = max(w, h)

# instantiate face models
face_detector = FaceDetector(model_path = str(MODEL_PATH / DETECT_MODEL),
                             delegate_path = args.delegate,
                             img_size = (target_dim, target_dim))
face_mesher = FaceMesher(model_path=str((MODEL_PATH / LANDMARK_MODEL)), delegate_path = args.delegate)
eye_mesher = EyeMesher(model_path=str((MODEL_PATH / EYE_MODEL)), delegate_path = args.delegate)

def draw_face_box(image, bboxes, landmarks, scores):
    for bbox, landmark, score in zip(bboxes.astype(int), landmarks.astype(int), scores):
        image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color=(255, 0, 0), thickness=2)
        landmark = landmark.reshape(-1, 2)

        score_label = "{:.2f}".format(score)
        (label_width, label_height), baseline = cv2.getTextSize(score_label,
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                fontScale=1.0,
                                                                thickness=2)
        label_btmleft = bbox[:2].copy() + 10
        label_btmleft[0] += label_width
        label_btmleft[1] += label_height
        cv2.rectangle(image, tuple(bbox[:2]), tuple(label_btmleft), color=(255, 0, 0), thickness=cv2.FILLED)
        cv2.putText(image, score_label, (bbox[0] + 5, label_btmleft[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
    return image

# detect single frame
def main(image):
    # pad image
    padded_size = [(target_dim - h) // 2, (target_dim - h + 1) // 2,
                (target_dim - w) // 2, (target_dim - w + 1) // 2]
    padded = cv2.copyMakeBorder(image.copy(),
                                *padded_size,
                                cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    padded = cv2.flip(padded, 3)

    # face detection
    bboxes_decoded, landmarks, scores = face_detector.inference(padded)

    mesh_landmarks_inverse = []
    r_vecs, t_vecs = [], []

    for i, (bbox, landmark) in enumerate(zip(bboxes_decoded, landmarks)):
        # landmark detection
        aligned_face, M, angel = face_detector.align(padded, landmark)
        mesh_landmark, mesh_scores = face_mesher.inference(aligned_face)
        mesh_landmark_inverse = face_detector.inverse(mesh_landmark, M)
        mesh_landmarks_inverse.append(mesh_landmark_inverse)

        # pose detection
        r_vec, t_vec = face_detector.decode_pose(landmark)
        r_vecs.append(r_vec)
        t_vecs.append(t_vec)

    # draw
    image_show = padded.copy()
    draw_face_box(image_show, bboxes_decoded, landmarks, scores)
    for i, (mesh_landmark, r_vec, t_vec) in enumerate(zip(mesh_landmarks_inverse, r_vecs, t_vecs)):
        mouth_ratio = get_mouth_ratio(mesh_landmark, image_show)
        left_box, right_box = get_eye_boxes(mesh_landmark, padded.shape)

        left_eye_img = padded[left_box[0][1]:left_box[1][1], left_box[0][0]:left_box[1][0]]
        right_eye_img = padded[right_box[0][1]:right_box[1][1], right_box[0][0]:right_box[1][0]]
        left_eye_landmarks, left_iris_landmarks = eye_mesher.inference(left_eye_img)
        right_eye_landmarks, right_iris_landmarks = eye_mesher.inference(right_eye_img)
        #cv2.rectangle(image_show, left_box[0], left_box[1], color=(255, 0, 0), thickness=2)
        #cv2.rectangle(image_show, right_box[0], right_box[1], color=(255, 0, 0), thickness=2)
        left_eye_ratio = get_eye_ratio(left_eye_landmarks, image_show, left_box[0])
        right_eye_ratio = get_eye_ratio(right_eye_landmarks, image_show, right_box[0])

        pitch, roll, yaw = get_face_angle(r_vec, t_vec)
        iris_ratio = get_iris_ratio(left_eye_landmarks, right_eye_landmarks)

        if mouth_ratio > 0.3:
            cv2.putText(image_show, "Yawning: Detected", (padded_size[2] + 70, padded_size[0] + 70),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 0, 0), thickness=2)
        else:
            cv2.putText(image_show, "Yawning: No", (padded_size[2] + 70, padded_size[0] + 70),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2)

        if left_eye_ratio < 0.2 and right_eye_ratio < 0.2:
            cv2.putText(image_show, "Eye: Closed", (padded_size[2] + 70, padded_size[0] + 100),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 0, 0), thickness=2)
        else:
            cv2.putText(image_show, "Eye: Open", (padded_size[2] + 70, padded_size[0] + 100),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2)

        if yaw > 15 and iris_ratio > 1.15:
            cv2.putText(image_show, "Face: Left",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 0, 0], thickness=2)
        elif yaw < -15 and iris_ratio < 0.85:
            cv2.putText(image_show, "Face: Right",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 0, 0], thickness=2)
        elif pitch > 30:
            cv2.putText(image_show, "Face: UP",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 0, 0], thickness=2)
        elif pitch < -13:
            cv2.putText(image_show, "Face: Down",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 0, 0], thickness=2)
        else:
            cv2.putText(image_show, "Face: Forward",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 255, 0], thickness=2)

    # remove pad
    image_show = image_show[padded_size[0]:target_dim - padded_size[1], padded_size[2]:target_dim - padded_size[3]]
    return image_show


# endless loop
while ret:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect single
    image_show = main(image)

    # put fps
    result = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)

    # display the result
    cv2.imshow('demo', result)

    ret, image = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

time.sleep(2)
cap.release()
cv2.destroyAllWindows()

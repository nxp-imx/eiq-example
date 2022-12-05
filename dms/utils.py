#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import math
import cv2
import numpy as np

def get_face_angle(rotation_vector, translation_vector):
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return pitch, roll, yaw

def get_hw_ratio(landmarks, points):
    landmarks = landmarks.astype(int)
    mouth_h = cv2.norm(landmarks[points[0]] - landmarks[points[1]])
    mouth_w = cv2.norm(landmarks[points[2]] - landmarks[points[3]])

    return  mouth_h/mouth_w

def get_mouth_ratio(landmarks, image):
    POINTS = (13, 14, 78, 308) #TOP, BOTTOM, LEFT, RIGHT
    ratio = get_hw_ratio(landmarks, POINTS)
    return ratio

def get_eye_ratio(landmarks, image, offsets=(0, 0)):
    POINTS = (12, 4, 0, 8) #TOP, BOTTOM, LEFT, RIGHT
    ratio = get_hw_ratio(landmarks, POINTS)

    return ratio

def get_iris_ratio(left_landmarks, right_landmarks):
    left_landmarks = left_landmarks.astype(int)
    right_landmarks = right_landmarks.astype(int)
    left = cv2.norm(left_landmarks[1] - left_landmarks[3])
    right = cv2.norm(right_landmarks[1] - right_landmarks[3])

    return left/right

def get_eye_boxes(landmarks, size, scale = 1.5):
    def get_box(landmarks, points, scale = 1.5):
        x_min, y_min = size[1], size[0]
        x_max, y_max = 0, 0
        for i in points:
            landmark = landmarks[i]
            x_min= min(landmark[0], x_min)
            y_min= min(landmark[1], y_min)
            x_max= max(landmark[0], x_max)
            y_max= max(landmark[1], y_max)
        left_box = ((x_min, y_min), (x_max, y_max))

        x_mid = (x_max + x_min) / 2
        y_mid = (y_max + y_min) / 2
        box_len = (x_max - x_min) * scale / 2
        x_min = int(max(x_mid - box_len, 0))
        x_max = int(min(x_mid + box_len, size[1]))
        y_min = int(max(y_mid - box_len, 0))
        y_max = int(min(y_mid + box_len, size[0]))

        return ((x_min, y_min), (x_max, y_max))

    LEFT_EYE_POINT  = [249, 263, 362, 373, 374, 380, 381, 382,
                       384, 385, 386, 387, 388, 390, 398, 466]
    RIGHT_EYE_POINT = [7, 33, 133, 144, 145, 153, 154, 155,
                       157, 158, 159, 160, 161, 163, 173, 246]

    left_box = get_box(landmarks, LEFT_EYE_POINT, scale)
    right_box = get_box(landmarks, RIGHT_EYE_POINT, scale)
    return left_box, right_box

def nms_oneclass(bbox, score, thresh = 0.4):
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = score.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)

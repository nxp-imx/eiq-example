#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import math
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from utils import nms_oneclass

FACE_MODEL_3D = np.array([
    (-165.0, 170.0, -135.0),  # left eye
    (165.0, 170.0, -135.0),   # right eye
    (0.0, 0.0, 0.0),  # Nose
    (0.0, -150, -110),  # mouth
    (-330.0, 100.0, -305.0),  # left face
    (330.0, 100.0, -305.0),  # right face
]) / 4.5  # 4.5 is scale factor

SCORE_THRESH = 0.9
MAX_FACE_NUM = 1
ANCHOR_STRIDES = [8, 16]
ANCHOR_NUM = [2, 6]

class FaceDetector:
    def __init__(self, model_path, delegate_path, img_size,
                 left_eye_pos=(0.38, 0.38),
                 target_width=192,
                 target_height=None):
        if(delegate_path):
            ext_delegate = [tflite.load_delegate(delegate_path)]
            self.interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=ext_delegate)
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]['index']
        self.input_shape = self.interpreter.get_input_details()[0]['shape'][1:3]

        self.outputs_idx = {}
        for output in self.interpreter.get_output_details():
            self.outputs_idx[output['name']] = output['index']

        self.anchors = self.create_anchors(self.input_shape)


        self.left_eye_pos = left_eye_pos
        self.target_width = target_width
        self.target_height = target_height
        if self.target_height is None:
            self.target_height = self.target_width

        self.size = img_size
        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")
        # Assuming no lens distortion
        self.dist_coeffs = np.zeros((4, 1))

    def inference(self, img):
        # convert to float32
        input_data = cv2.resize(img, tuple(self.input_shape)).astype(np.float32)
        input_data = (input_data - 128.0) / 128.0
        input_data = np.expand_dims(input_data, axis=0)

        # invoke
        self.interpreter.set_tensor(self.input_idx, input_data)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.outputs_idx['classificators']).squeeze()
        scores = 1 / (1 + np.exp(-scores))
        bboxes = self.interpreter.get_tensor(self.outputs_idx['regressors']).squeeze()

        bboxes_decoded, landmarks, scores = self.decode(scores, bboxes)
        bboxes_decoded *= img.shape[0]
        landmarks *= img.shape[0]

        if len(bboxes_decoded) != 0:
            keep_mask = nms_oneclass(bboxes_decoded, scores)  # np.ones(pred_bbox.shape[0]).astype(bool)
            bboxes_decoded = bboxes_decoded[keep_mask]
            landmarks = landmarks[keep_mask]
            scores = scores[keep_mask]

            return bboxes_decoded, landmarks, scores
        else:
            return np.array([]), np.array([]), np.array([])

    def decode(self, scores, bboxes):
        w, h = self.input_shape
        top_score = np.sort(scores)[-MAX_FACE_NUM]
        cls_mask = scores >= max(SCORE_THRESH, top_score)
        if cls_mask.sum() == 0:
            return np.array([]), np.array([]), np.array([])

        scores = scores[cls_mask]
        bboxes = bboxes[cls_mask]
        bboxes_anchors = self.anchors[cls_mask]

        bboxes_decoded = bboxes_anchors.copy()
        bboxes_decoded[:, 0] += bboxes[:, 1]  # row
        bboxes_decoded[:, 1] += bboxes[:, 0]  # columns
        bboxes_decoded[:, 0] /= h
        bboxes_decoded[:, 1] /= w

        pred_w = bboxes[:, 2] / w
        pred_h = bboxes[:, 3] / h

        topleft_x = bboxes_decoded[:, 1] - pred_w * 0.5
        topleft_y = bboxes_decoded[:, 0] - pred_h * 0.5
        btmright_x = bboxes_decoded[:, 1] + pred_w * 0.5
        btmright_y = bboxes_decoded[:, 0] + pred_h * 0.5

        pred_bbox = np.stack([topleft_x, topleft_y, btmright_x, btmright_y], axis=-1)

        # decode landmarks
        landmarks = bboxes[:, 4:]
        landmarks[:, 1::2] += bboxes_anchors[:, 0:1]
        landmarks[:, ::2] += bboxes_anchors[:, 1:2]
        landmarks[:, 1::2] /= h
        landmarks[:, ::2] /= w

        return pred_bbox, landmarks, scores

    @classmethod
    def create_anchors(cls, input_shape):
        w, h = input_shape
        anchors = []
        for s, a_num in zip(ANCHOR_STRIDES, ANCHOR_NUM):
            gridCols = (w + s - 1) // s
            gridRows = (h + s - 1) // s
            x, y = np.meshgrid(np.arange(gridRows), np.arange(gridCols))
            x, y = x[..., None], y[..., None]
            anchor_grid = np.concatenate([y, x], axis=-1)
            anchor_grid = np.tile(anchor_grid, (1, 1, a_num))
            anchor_grid = s * (anchor_grid.reshape(-1, 2) + 0.5)
            anchors.append(anchor_grid)
        return np.concatenate(anchors, axis=0)

    def align(self, image, landmarks):
        landmarks = landmarks.astype(int).reshape(-1, 2)

        # get left and right eye
        left_eye = landmarks[1]
        right_eye = landmarks[0]

        # computer angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the location of right/left eye in new image
        right_eye_pos = 1 - self.left_eye_pos[0]

        # get the scale based on the distance
        dist = np.sqrt(dY ** 2 + dX ** 2)
        desired_dist = (right_eye_pos - self.left_eye_pos[0])
        desired_dist *= self.target_width
        scale = desired_dist / (dist + 1e-6)

        # get the center of eyes
        eye_center = (left_eye + right_eye) // 2

        # get transformation matrix
        M = cv2.getRotationMatrix2D((int(eye_center[0]), int(eye_center[1])), angle, scale)

        # align the center
        tX = self.target_width * 0.5
        tY = self.target_height * self.left_eye_pos[1]
        M[0, 2] += (tX - eye_center[0])
        M[1, 2] += (tY - eye_center[1])  # update translation vector

        # apply affine transformation
        dst_size = (self.target_width, self.target_height)
        output = cv2.warpAffine(image, M, dst_size, flags=cv2.INTER_CUBIC)
        return output, M, angle

    @staticmethod
    def inverse(mesh_landmark, M):
        M_inverse = cv2.invertAffineTransform(M)
        px = (M_inverse[0, 0] * mesh_landmark[:, 0:1] + M_inverse[0, 1] * mesh_landmark[:, 1:2] + M_inverse[0, 2])
        py = (M_inverse[1, 0] * mesh_landmark[:, 0:1] + M_inverse[1, 1] * mesh_landmark[:, 1:2] + M_inverse[1, 2])
        mesh_landmark_inverse = np.concatenate([px, py, mesh_landmark[:, 2:]], axis=-1)
        return mesh_landmark_inverse


    def decode_pose(self, landmarks):
        landmarks = landmarks.astype(float).reshape(-1, 2)
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            FACE_MODEL_3D, landmarks, self.camera_matrix, self.dist_coeffs)

        return rotation_vector, translation_vector

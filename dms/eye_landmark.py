#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class EyeMesher:
    EYE_KEY_NUM = 71
    IRIS_KEY_NUM = 5

    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]['index']
        self.input_shape = self.interpreter.get_input_details()[0]['shape'][1:3]

        outputs_idx_tmp = {}
        for output in self.interpreter.get_output_details():
            outputs_idx_tmp[output['name']] = output['index']
        self.outputs_idx = {'eye': outputs_idx_tmp['output_eyes_contours_and_brows:0'],
                'iris': outputs_idx_tmp['output_iris:0']}

    def inference(self, image):
        h, w = self.input_shape

        image_ = cv2.resize(image, tuple(self.input_shape)).astype(np.float32)
        image_ = (image_) / 255.0
        if len(image_.shape) < 4:
            image_ = image_[None, ...]

        # invoke
        self.interpreter.set_tensor(self.input_idx, image_)
        self.interpreter.invoke()
        eye_landmarks = self.interpreter.get_tensor(self.outputs_idx['eye'])
        iris_landmarks = self.interpreter.get_tensor(self.outputs_idx['iris'])

        # postprocessing
        eye_landmarks = eye_landmarks.reshape(self.EYE_KEY_NUM, 3)
        eye_landmarks[:, 0] /= w
        eye_landmarks[:, 1] /= h
        eye_landmarks[:, 0] *= image.shape[1]
        eye_landmarks[:, 1] *= image.shape[0]

        iris_landmarks = iris_landmarks.reshape(self.IRIS_KEY_NUM, 3)
        iris_landmarks[:, 0] /= w
        iris_landmarks[:, 1] /= h
        iris_landmarks[:, 0] *= image.shape[1]
        iris_landmarks[:, 1] *= image.shape[0]

        return eye_landmarks, iris_landmarks



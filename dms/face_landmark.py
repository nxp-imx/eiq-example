#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import math
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

class FaceMesher:
    FACE_KEY_NUM = 468

    def __init__(self, model_path, delegate_path):
        if(delegate_path):
            ext_delegate = [tflite.load_delegate(delegate_path)]
            self.interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=ext_delegate)
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]['index']
        self.input_shape = self.interpreter.get_input_details()[0]['shape'][1:3]

        outputs_idx_tmp = {}
        for output in self.interpreter.get_output_details():
            outputs_idx_tmp[output['name']] = output['index']
        self.outputs_idx = {'landmark': outputs_idx_tmp['conv2d_20'],
                            'score': outputs_idx_tmp['conv2d_30']}

    def inference(self, img):
        h, w = self.input_shape
        input_data = cv2.resize(img, tuple(self.input_shape)).astype(np.float32)
        input_data = (input_data - 128.0) / 128.0
        input_data = np.expand_dims(input_data, axis=0)

        # invoke
        self.interpreter.set_tensor(self.input_idx, input_data)
        self.interpreter.invoke()
        landmarks = self.interpreter.get_tensor(self.outputs_idx['landmark'])
        scores = self.interpreter.get_tensor(self.outputs_idx['score'])

        # postprocessing
        landmarks = landmarks.reshape(self.FACE_KEY_NUM, 3)
        landmarks[:, 0] /= w
        landmarks[:, 1] /= h
        landmarks[:, 0] *= img.shape[1]
        landmarks[:, 1] *= img.shape[0]
        score = 1.0 / (1.0 + math.exp(-scores.ravel()[0]))

        return landmarks, score


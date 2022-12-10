#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import time


class Facenet(object):
    def __init__(self, model_path, delegate_path):
        if(delegate_path):
            ext_delegate = [tflite.load_delegate(delegate_path)]
            self.interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=ext_delegate)
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]

    def _pre_processing(self, img):
        x = cv2.resize(img, self.input_shape)
        mean = np.mean(x, dtype="float32")
        std = np.std(x, dtype="float32")
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))

        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return np.expand_dims(y, axis=0)

    def get_embeddings(self, face):
        input_data = self._pre_processing(face)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        embeddings = self.interpreter.get_tensor(self.output_details[0]['index'])
        return embeddings[0]

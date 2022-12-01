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

import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import time


class Facenet(object):
    def __init__(self, model_path):
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

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
import time
import argparse
from hand_tracker import HandTracker

PALM_MODEL_PATH = "../vela_models/palm_detection_builtin_256_integer_quant_vela.tflite"
LANDMARK_MODEL_PATH = "../vela_models/hand_landmark_3d_256_integer_quant_vela.tflite"
ANCHORS_PATH = "anchors.csv"

def draw_landmarks(points, frame):
    #        8   12  16  20
    #        |   |   |   |
    #        7   11  15  19
    #    4   |   |   |   |
    #    |   6   10  14  18
    #    3   |   |   |   |
    #    |   5---9---13--17
    #    2    \         /
    #     \    \       /
    #      1    \     /
    #       \    \   /
    #        ------0-
    connections = [
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (0, 5), (5, 9), (9, 13), (13, 17), (0, 9), (0, 13)
    ]
    connections += [(0, 17), (17, 18), (18, 19), (19, 20), (0, 1), (1, 2), (2, 3), (3, 4)]

    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), 2)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input',
    default='/dev/video0',
    help='input to be classified')
args = parser.parse_args()
capture = cv2.VideoCapture(args.input)
ret, frame = capture.read()
if (frame is None):
    print("Can't read frame from source file ", args.input)
    exit(0)

detector = HandTracker(PALM_MODEL_PATH, LANDMARK_MODEL_PATH, ANCHORS_PATH, box_shift=0.2, box_enlarge=1.3)

while ret:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)
    draw_landmarks(points, frame)
    cv2.imshow("hand", frame)

    ret, frame = capture.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

time.sleep(2)
capture.release()
cv2.destroyAllWindows()

#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import os


def _l2_normalization(x, y):
    #Compute the L2 norm of ( x - y )
    return np.linalg.norm(x - y)

def _cosine_similarity(x, y):
    #Compute the cosine of the angle between x and y
    return np.sqrt(np.sum(np.square(np.subtract(x, y))))

def _sort_dict(x):
    return sorted(x.items(), key=lambda d:d[1])

class FaceDatabase(object):
     def __init__(self, threshold = 21, db_file = "database.npy"):
         self.threshold = threshold
         self.db_file = db_file
         try:
             self.database = np.load(db_file, allow_pickle=True).item()
         except FileNotFoundError:
             self.database = {}

     def find_name(self, y):
         scores = {}
         for name in self.database:
             scores[name] = []
             for x in self.database[name]:
                 scores[name].append(_l2_normalization(x, y))
             scores[name] = np.mean(scores[name])

         if (scores == {}):
             return "Unknown"

         smallest = _sort_dict(scores)[0]
         if smallest[1] < self.threshold:
             return smallest[0]
         else:
             return "Unknown"

     def add_name(self, name: str, embeddings: list):
         if (name == "" or len(embeddings) != 512):
             return False

         if (name in self.database):
             self.database[name].append(embeddings)
         else:
             self.database[name] = [embeddings]
         np.save(self.db_file, self.database)
         return True

     def del_name(self, name: str):
         if (name == "" or name not in self.database):
             return False

         self.database.pop(name)
         np.save(self.db_file, self.database)
         return True

     def get_names(self):
         return list(self.database.keys())

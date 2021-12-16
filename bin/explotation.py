
import numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argsort, shape 
from scipy import interpolate,ndimage


image = np.load("arraySave/image.npy")
boundingBox = np.load("arraySave/boundingBox.npy",allow_pickle=True )
label_image = np.load("arraySave/label_image.npy")


# Triage des pierre par Taille en pixel
num_pierre = len(boundingBox)
ListTaillePierre = []
for pierre in range(num_pierre):
    boundingBoxPierre = (boundingBox[pierre][0],boundingBox[pierre][1])
    taillePierre = len(np.argwhere(label_image[boundingBoxPierre] == 1))
    ListTaillePierre.append(taillePierre)

Index_sort = numpy.argsort(-np.asarray(ListTaillePierre))


for pierre in range(num_pierre):
    pierre_trie = Index_sort[pierre]
    boundingBoxPierre = (boundingBox[pierre_trie][0],boundingBox[pierre_trie][1])
    a = image[boundingBoxPierre]
    plt.figure()
    plt.imshow(a)
    plt.show()

plt.figure()
print((boundingBox[21][0],boundingBox[21][1]))
plt.imshow(label_image[222:263,114:156])
plt.show()

import cv2
from matplotlib.colors import BoundaryNorm
import numpy as np
import matplotlib.pyplot as plt 
from scipy import interpolate,ndimage
from math import sqrt


# Chemin de la Texture
Texture = cv2.imread("../Textures/freeTexture3.png")
Texture = cv2.cvtColor(Texture, cv2.COLOR_BGR2RGB)
Taille_chemin = 1

###### Segmentation

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = Texture.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 2
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()
segmented_image = centers[labels.flatten()]

# reshape back to the original image dimension
segmented_image = segmented_image.reshape(Texture.shape)

###### Labelisation des pierre
label_image = -labels.reshape(512,512)+1

# Separation des differentes pierres
pierre_segmenter, num_pierre = ndimage.label(label_image)

# BoundingBox
boundingBox = ndimage.find_objects(pierre_segmenter)

### Triage des pierre selon leur taille

# Triage des pierre par Taille en pixel
num_pierre = len(boundingBox)
ListTaillePierre = []
for pierre in range(num_pierre):
    boundingBoxPierre = (boundingBox[pierre][0],boundingBox[pierre][1])
    taillePierre = len(np.argwhere(label_image[boundingBoxPierre] == 1))
    ListTaillePierre.append(taillePierre)

Index_sort = np.argsort(-np.asarray(ListTaillePierre))

###### Sauvegarde Des donnes
np.save("arraySave/image",Texture)
np.save("arraySave/boundingBox",boundingBox)
np.save("arraySave/label_image",label_image)
np.save("arraySave/PierreSorted",Index_sort)

print("Nombre de pierre trouvé dans l'image:", num_pierre)
print("taille des pierre trié:",end="")
print(np.take_along_axis(np.asarray(ListTaillePierre), Index_sort, axis=0))


#### Retour Graphique

plt.figure()
plt.subplot(2,2,1)
a = plt.imshow(segmented_image,origin='lower')
plt.savefig("../Resultat/Segmentation/segmentation.png")
plt.subplot(2,2,2)
plt.imshow(Texture,origin='lower')
plt.subplot(2,2,3)
plt.imshow(pierre_segmenter,origin='lower')
plt.show()
plt.savefig(a,"../Resultat/Segmentation/segmentation.png")

plt.figure()
for pierre in range(num_pierre):
    plt.subplot(round(sqrt(num_pierre))+1,round(sqrt(num_pierre))+1,pierre+1)
    pierre_trie = Index_sort[pierre]
    boundingBoxPierre = (boundingBox[pierre_trie][0],boundingBox[pierre_trie][1])
    a = Texture[boundingBoxPierre]
    plt.imshow(a)
plt.show()

plt.savefig("../Resultat/Segmentation/Pierre.png")


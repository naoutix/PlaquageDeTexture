import cv2
from matplotlib.colors import BoundaryNorm
import numpy as np
import matplotlib.pyplot as plt 
from scipy import interpolate,ndimage

# Chemin de la Texture
Texture = cv2.imread("C:/Users/leoar/Documents/n7/3A/APP/PlaquageDeTexture/Textures/freeTexture3.png")
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
plt.imshow(segmented_image,origin='lower')
plt.savefig("C:/Users/leoar/Documents/n7/3A/APP/PlaquageDeTexture/Resultat/Segmentation/segmented_image.png")
plt.show()

###### Labelisation des pierre
label_image = -labels.reshape(512,512)+1
pierre_segmenter, num_pierre = ndimage.label(label_image)
plt.imshow(pierre_segmenter,origin='lower')
plt.show()
boundingBox = ndimage.find_objects(pierre_segmenter)

np.save("arraySave/boundingBox",boundingBox)
np.save("arraySave/label_image",label_image)
plt.figure()
print(num_pierre)
print(len(boundingBox))
for pierre in range(num_pierre):
    #plt.subplot(round(np.sqrt(num_pierre))+1,round(np.sqrt(num_pierre))+1,pierre+1)
    a=segmented_image[boundingBox[pierre]]
    #mask = label_image[boundingBox[pierre+6]]
    plt.figure()
    plt.imshow(a)
    plt.show()
    print(pierre)
    #np.savetxt("arraySave/pierre_1",mask,fmt="%d")

#plt.imshow(pierre_segmenter,origin='lower')
plt.show()


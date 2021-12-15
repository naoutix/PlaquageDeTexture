import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage

## Reference du background & texture
patch_background_elem = cv2.imread("C:/Users/leoar/Documents/n7/3A/APP/PlaquageDeTexture/Textures/freeTexture2.png")
Texture = cv2.imread("C:/Users/leoar/Documents/n7/3A/APP/PlaquageDeTexture/Textures/freeTexture3.png")
Taille_chemin = 200

######## Taille de patch de background ########


patch_background_elem = cv2.cvtColor(patch_background_elem, cv2.COLOR_BGR2RGB)
Texture = cv2.cvtColor(Texture, cv2.COLOR_BGR2RGB)

L = 8 #Largeur
l = 8 #Longeur
[l_elem,L_elem,nb_canaux] = patch_background_elem.shape
max_longeur_chemin = 10

###### Creation du background


patch_colonne = np.copy(patch_background_elem)
for i in range(L):
    patch_colonne = np.concatenate((patch_colonne,patch_background_elem),axis=0)

environnement = np.copy(patch_colonne)
for i in range(l):
    environnement = np.concatenate((environnement,patch_colonne),axis=1)

###### Selection des points


plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
plt.imshow(environnement)
points = plt.ginput(max_longeur_chemin)   #Donnee Points
pointx = np.asarray(points)[:,0]
pointy = np.asarray(points)[:,1]
plt.scatter(pointx,pointy,color='r')

###### Interpolation du chemin 


# Spline Parametrique (NbPoints > 4)
nbPoint = 50
Points_parametrisation = np.linspace(0, 1.01, nbPoint)
tck, u = interpolate.splprep([pointx, pointy], s=0)
chemin = interpolate.splev(Points_parametrisation, tck)
plt.plot(chemin[0],chemin[1])
plt.savefig("C:/Users/leoar/Documents/n7/3A/APP/PlaquageDeTexture/Resultat/Interpolation/draw_on_image_01.png")
plt.show()
plt.close()

# Implentation du chemin avec 
x=np.arange(-Taille_chemin,Taille_chemin+1,1)
X,Y = np.meshgrid(x,x)
cercle = ~(X*X+Y*Y < Taille_chemin*Taille_chemin)
cercle = cercle.astype(int)
cercle3 = np.zeros((2*Taille_chemin+1,2*Taille_chemin+1,3))
cercle3[:,:,0] = cercle
cercle3[:,:,1] = cercle
cercle3[:,:,2] = cercle
[maskl,maskL,_] = environnement.shape
mask = np.ones((maskl,maskL))
for i in range(nbPoint):
    yc = round(chemin[0][i])
    xc = round(chemin[1][i])
    #environnement[xc-Taille_chemin:xc+Taille_chemin+1,yc-Taille_chemin:yc+Taille_chemin+1,:] = environnement[xc-Taille_chemin:xc+Taille_chemin+1,yc-Taille_chemin:yc+Taille_chemin+1,:]*cercle3
    mask[xc-Taille_chemin:xc+Taille_chemin+1,yc-Taille_chemin:yc+Taille_chemin+1]=mask[xc-Taille_chemin:xc+Taille_chemin+1,yc-Taille_chemin:yc+Taille_chemin+1]*cercle
plt.imshow(mask,cmap='gray', vmin = 0, vmax = 1)
plt.savefig("C:/Users/leoar/Documents/n7/3A/APP/PlaquageDeTexture/Resultat/Interpolation/chemin.png")
plt.show()




###### Segmentation


plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)

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

###### Incrustation chemin

image = -labels.reshape(512,512)+1
pierre_segmenter, num_pierre = ndimage.label(image)
plt.imshow(pierre_segmenter,origin='lower')
plt.show()

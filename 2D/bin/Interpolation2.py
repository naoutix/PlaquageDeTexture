import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,sparse

from plaquage.Interpolation import create_background,selection_chemin2,largeur_R_chemin2
## Reference du background
patch_background_elem = cv2.imread("texture_finale.png")
patch_background_elem = cv2.cvtColor(patch_background_elem, cv2.COLOR_BGR2RGB)
Taille_chemin = 5
chemins = cv2.imread("routes.png")
chemins = cv2.cvtColor(chemins, cv2.COLOR_BGR2RGB)
L = 0 #Largeur
l = 0 #Longeur
max_longeur_chemin = 200
nbPoint = 200
######## Taille de patch de background ########
environnement = create_background(patch_background_elem,l,L)
###### Selection des points
points,fig,ax = selection_chemin2(chemins,environnement)
###### Interpolation du chemin 
chemin = points
###### Creation Mask
mask_bsr = largeur_R_chemin2(Taille_chemin,environnement,chemin)
sparse.save_npz("arraySave/MaskChemin",mask_bsr)
cv2.imwrite("arraySave/environnement.jpg", environnement)

### Retour graphique 
pointx = np.asarray(points)[:,0]
pointy = np.asarray(points)[:,1]
ax.scatter(pointx,pointy,color='r')
ax2 =plt.subplot(1,2,2)
ax2.imshow(mask_bsr.toarray(),cmap='gray', vmin = 0, vmax = 1,origin='lower')
ax2.set_title("Masque de placement des pierres")
plt.savefig("../Resultat/Interpolation/chemin.png")
plt.show()


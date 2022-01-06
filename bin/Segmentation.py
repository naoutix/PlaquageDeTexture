import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append 
from math import sqrt

from plaquage.Segmentation import segmentation,sep_box_pierre

# Parametre
Repetable  = True
Edge_cutting = True
Reverse = True

# Chemin de la Texture
Texture = cv2.imread("../Textures/freeTexture3.png")
Texture = cv2.cvtColor(Texture, cv2.COLOR_BGR2RGB)
Taille_chemin = 1
nbligne,nbcol,canaux = Texture.shape

###### Segmentation
label_image,image_valeur,segmented_image = segmentation(Texture,Reverse,Repetable)
Pierres,Pierres_mask,ListTaillePierre,num_pierre,pierre_segmenter = sep_box_pierre(label_image,image_valeur,Edge_cutting)

###### Sauvegarde Des donnes
np.save("arraySave/image",Texture)
np.save("arraySave/PierreMask",Pierres_mask)
np.save("arraySave/PierreSorted",Pierres)
np.save("arraySave/ListTaillePierre",ListTaillePierre)

print("Nombre de pierre trouvé dans l'image:", num_pierre)
print("taille des pierre trié:",end="")
print(ListTaillePierre)


# #### Retour Graphique

fig = plt.figure(figsize=(13, 7), dpi=150)
ax = plt.subplot(2,2,1)
ax.imshow(image_valeur ,origin='lower')
ax.set_title('Image: ' + 'freeTexture3.png')
ax2 = plt.subplot(2,2,2)
ax2.imshow(segmented_image,origin='lower')
ax2.set_title('Image Segmenter par:' + 'kmeans')
ax3 = plt.subplot(2,1,2)
ax3.imshow(pierre_segmenter,origin='lower',vmin=0,vmax=num_pierre)
ax3.set_title('Separation des pierres')
fig.suptitle('Resultat Segmentation')
plt.savefig("../Resultat/Segmentation/segmentation.png",dpi=150)


fig2 = plt.figure(figsize=(13, 7), dpi=150)
fig2.suptitle("Pierres trouvé")
for pierre in range(num_pierre):
    ax = plt.subplot(round(sqrt(num_pierre))+1,round(sqrt(num_pierre))+1,pierre+1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(pierre+1,loc='left')
    plt.imshow(Pierres[pierre])
plt.savefig("../Resultat/Segmentation/Pierres_Rep_"+str(int(Repetable))+"_Edge_"+str(int(Edge_cutting))+".png",dpi=150)

### Affichage
plt.show()

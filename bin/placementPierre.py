import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,misc,sparse,stats
from math import inf,sqrt
import skfmm
import time
from plaquage.Interpolation import limite_changer

### Retour graphique
def update(ax1,ax2,ax3,ax4,ax5,ax6,mask_array,environnement,densite_graph,sorted_point,nb_iter,Poisson_grap,res):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    ### Lim
    ax3.set_xlim(0,100)
    ax3.set_ylim(0,10)
    ax4.set_xlim(0,100)
    ax4.set_ylim(0,num_pierre_max/2)
    ax5.set_xlim(0,100)
    ax5.set_ylim(0,100)
    ## Label
    ax3.set_xlabel("Nb_iteration")
    ax3.set_ylabel("Nb_pierre place")
    ax4.set_xlabel("Nb_iteration")
    ax4.set_ylabel("lambda")
    ax5.set_xlabel("Nb_iteration")
    ax5.set_ylabel("%")
    ax6.set_xlabel("Fast_Marching")
    ### Plot
    ax1.imshow(mask_array,cmap="gray",origin="lower")
    ax1.scatter(sorted_point[1],sorted_point[0],s=1)
    ax2.imshow(environnement,origin="lower")
    ax3.plot(nb_iter,nb_pierre_place)
    ax4.plot(nb_iter,Poisson_grap)
    ax5.plot(nb_iter,densite_graph)
    ax6.imshow(res,cmap="gray",origin="lower")
    plt.pause(0.1)

## Retour Graphique
plt.ion()
fig = plt.figure(figsize=(10, 5), dpi=150) 
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,4)
ax4 = plt.subplot(2,3,5)
ax5 = plt.subplot(2,3,6)
ax6 = plt.subplot(2,3,3)

### Importation des données
image = np.load("arraySave/image.npy")
mask = sparse.load_npz("arraySave/MaskChemin.npz")
environnement = cv2.imread("arraySave/environnement.jpg")
ListeTaillePierre = np.load("arraySave/ListTaillePierre.npy" )
Pierre_mask = np.load("arraySave/PierreMask.npy",allow_pickle=True)
Pierre_sorted = np.load("arraySave/PierreSorted.npy",allow_pickle=True)

mask_array = mask.toarray()
#res = ndimage.gaussian_filter(mask.toarray(),sigma =10,order=2)
[maskl,maskL] = mask_array.shape

### Algorithme de placement des pierre
k_max=0                                ### iteration maximum
Poisson = 1                            ### lambda de départ
alpha = 0.9                            ### Coefficient de décroissance
Taille_chemin =200                     ### Taille du chemin
T = 10                                 ### Temperature initial
num_pierre_max = len(Pierre_mask)      ###  Nombre de pierre
pas = np.argwhere(mask_array == 1).shape[0]/Taille_chemin ### Pas
print(len(np.argwhere(mask_array==1)))
print(len(np.argwhere(mask_array==0)))
stop=False
k = 0

### Suivi
densite_div = len(np.argwhere(mask_array == 1))
densite_k = densite_div
nb_iter = []
nb_pierre_place = []
Poisson_grap = []
densite_graph = []

## Init
res = skfmm.distance(mask_array,dx=1)
sort = np.argsort(-res,axis=None)
index = np.unravel_index(sort[0:int(pas):int(pas/T)], res.shape)
index = np.array(index).transpose()
index  = index.tolist()
sorted_point = np.unravel_index(sort[0:int(pas):int(pas/T)],res.shape)

update(ax1,ax2,ax3,ax4,ax5,ax6,mask_array,environnement,densite_graph,sorted_point,nb_iter,Poisson_grap,res)

while not stop:
    echec = 0
    ### Meilleur placement trouver par fast_marching
    ### Placement de T pierre
    for i in index:
        ## Selection de la pierre
        num_pierre = stats.poisson.rvs(Poisson)
        if (num_pierre >= num_pierre_max):
            num_pierre = num_pierre_max -1
        mask_pierre = Pierre_mask[num_pierre]
        pierre = Pierre_sorted[num_pierre]

        ## Dimension pierre
        [l_pierre,L_pierre] = mask_pierre.shape
        limite = [slice(i[0]-int(l_pierre/2),i[0]+int(l_pierre/2+0.5),None),slice(i[1]-int(L_pierre/2),i[1]+int(L_pierre/2+0.5))]
        pierre_limite = [slice(0,l_pierre,None),slice(0,L_pierre,None)]
        limite,pierre_limite = limite_changer(limite,pierre_limite,maskl,maskL)
        ## Verification chevauchement pierre
        diff_mask = mask_array[limite]-mask_pierre[pierre_limite].astype(int)
        if diff_mask.min() >= 0:
        ### Incrustation pierre
            environnement[limite][mask_pierre[pierre_limite]]= pierre[pierre_limite][mask_pierre[pierre_limite]]
            ## Maj masque
            mask_array[limite][mask_pierre[pierre_limite]] = 0
            limite2 = tuple([slice(max(limite[0].start-Taille_chemin,0),min(limite[0].stop+Taille_chemin,maskl),None),slice(min(limite[1].start-Taille_chemin,0),max(limite[1].stop+Taille_chemin,maskL),None)])
            res[limite2] = np.minimum(skfmm.distance(mask_array[limite2],dx=1),res[limite2])
            densite_k = densite_k-ListeTaillePierre[num_pierre]
        else:
            echec = echec+1
    if (echec > T/1.5):
        Poisson = Poisson + 1
    else:
        if(Poisson > 1 and (echec < (T/2))):
            Poisson = Poisson - 1
    if T > num_pierre_max/2:
        stop = True

    ## Maj
    k = k+1
    nb_iter.append(k)
    nb_pierre_place.append(T-echec)
    Poisson_grap.append(Poisson)
    densite_graph.append((densite_k/densite_div) *100)
    sorted_point = np.unravel_index(sort[0:int(pas):int(pas/T)],res.shape)
    update(ax1,ax2,ax3,ax4,ax5,ax6,mask_array,environnement,densite_graph,sorted_point,nb_iter,Poisson_grap,res)
    if T-echec != 0:
        sort = np.argsort(-res,axis=None)
        index = np.unravel_index(sort[0:int(pas):int(pas/T)], res.shape)
        index = np.array(index).transpose()
        index  = index.tolist()
    
plt.savefig("../Resultat/placementPierre/courbes.png",dpi=150)

plt.figure()
plt.imshow(environnement,origin="lower")
plt.savefig("../Resultat/placementPierre/final.png",dpi=150)
plt.show()

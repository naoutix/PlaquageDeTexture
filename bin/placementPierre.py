import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,misc,sparse,stats
from math import inf,sqrt
import skfmm
import time

### Retour graphique
def update(ax1,ax2,ax3,ax4,ax5,mask_array,environnement,densite_graph,sorted_point,nb_iter):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.set_xlim(0,100)
    ax3.set_ylim(0,10)
    ax4.set_xlim(0,100)
    ax4.set_ylim(0,13)
    ax5.set_xlim(0,100)
    ax5.set_ylim(0,100)
    ax3.set_xlabel("Nb_iteration")
    ax3.set_ylabel("Nb_pierre place")
    ax4.set_xlabel("Nb_iteration")
    ax4.set_ylabel("T")
    ax5.set_xlabel("Nb_iteration")
    ax5.set_ylabel("%")
    im1 = ax1.imshow(mask_array,cmap="gray",origin="lower")
    ax1.scatter(sorted_point[1],sorted_point[0],s=1)
    im2 = ax2.imshow(environnement,origin="lower")
    im3 = ax3.plot(nb_iter,nb_pierre_place)
    im4 = ax4.plot(nb_iter,T_grap)
    im5 = ax5.plot(nb_iter,densite_graph)
    plt.pause(0.1)


### Importation des données
image = np.load("arraySave/image.npy")
mask = sparse.load_npz("arraySave/MaskChemin.npz")
environnement = cv2.imread("arraySave/environnement.jpg")
ListeTaillePierre = np.load("arraySave/ListTaillePierre.npy" )
Pierre_mask = np.load("arraySave/PierreMask.npy",allow_pickle=True)
Pierre_sorted = np.load("arraySave/PierreSorted.npy",allow_pickle=True)

mask_array = -mask.toarray()+1
#res = ndimage.gaussian_filter(mask.toarray(),sigma =10,order=2)
[maskl,maskL] = mask_array.shape


### Algorithme de placement des pierre
k_max=0
T = 1
alpha = 0.9
num_pierre_max = len(Pierre_mask)
stop=False
k = 0
nb_try = 1
nb_pierre_pass = 10

pas = maskl*maskL/(4*200)
densite_div = len(np.argwhere(mask_array == 1))
densite_k = densite_div
nb_iter = []
nb_pierre_place = []
T_grap = []
densite_graph = []
## Init
res = skfmm.distance(mask_array,dx=1)
sort = np.argsort(-res,axis=None)
index = np.unravel_index(sort[0:int(pas):int(pas/nb_pierre_pass)], res.shape)
index = np.array(index).transpose()
index  = index.tolist()


## Retour Graphique
plt.ion()
fig = plt.figure(figsize=(10, 5), dpi=150) 
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,4)
ax4 = plt.subplot(2,3,5)
ax5 = plt.subplot(2,3,6)
sorted_point = np.unravel_index(sort[0:int(pas):int(pas/nb_pierre_pass)],res.shape)
update(ax1,ax2,ax3,ax4,ax5,mask_array,environnement,densite_graph,sorted_point,nb_iter)


while not stop:
    echec = 0
    ### Meilleur placement trouver par fast_marching
    ### Placement de T pierre
    for i in index:
        ## Selection de la pierre
        num_pierre = stats.poisson.rvs(T)
        if (num_pierre >= num_pierre_max):
            num_pierre = num_pierre_max -1
        mask_pierre = Pierre_mask[num_pierre]
        pierre = Pierre_sorted[num_pierre]

        ## Dimension pierre
        [l_pierre,L_pierre] = mask_pierre.shape
        xmin = i[0]-int(l_pierre/2)
        xmax = i[0]+int(l_pierre/2+0.5)
        ymin = i[1]-int(L_pierre/2)
        ymax = i[1]+int(L_pierre/2+0.5)
        xpmin = 0
        xpmax = l_pierre
        ypmin = 0
        ypmax = L_pierre
        if (xmin < 0):
            xpmin = -xmin
            xmin = 0
        if (xmax >= maskl):
            xpmax = l_pierre-1 - (xmax - maskl)
            xmax = maskl-1
        if (ymin < 0):
            ypmin = -ymin
            ymin = 0
        if (ymax  >= maskL):
            ypmax = L_pierre-1 - (ymax - maskL)
            ymax = maskL-1

        ## Verification chevauchement pierre
        diff_mask = mask_array[xmin:xmax,ymin:ymax]-mask_pierre[xpmin:xpmax,ypmin:ypmax]
        if diff_mask.min() >= 0:
        ### Incrustation pierre
            for i in range(3):
                environnement[xmin:xmax,ymin:ymax,i]=environnement[xmin:xmax,ymin:ymax,i]*(-mask_pierre[xpmin:xpmax,ypmin:ypmax]+1)
            environnement[xmin:xmax,ymin:ymax]= environnement[xmin:xmax,ymin:ymax]+pierre[xpmin:xpmax,ypmin:ypmax]

            ## Maj masque
            mask_array[xmin:xmax,ymin:ymax] = mask_array[xmin:xmax,ymin:ymax]*(-mask_pierre[xpmin:xpmax,ypmin:ypmax]+1)
            densite_k = densite_k-ListeTaillePierre[num_pierre]
        else:
            echec = echec+1
    if (echec > nb_pierre_pass/1.5):
        T = T + 1
    else:
        if(T > 1 and (echec < (nb_pierre_pass/2))):
            T = T - 1
    if T > 13:
        stop = True

    ## Maj
    k = k+1
    nb_iter.append(k)
    nb_pierre_place.append(nb_pierre_pass-echec)
    T_grap.append(T)
    densite_graph.append((densite_k/densite_div) *100)
    sorted_point = np.unravel_index(sort[0:int(pas):int(pas/nb_pierre_pass)],res.shape)
    update(ax1,ax2,ax3,ax4,ax5,mask_array,environnement,densite_graph,sorted_point,nb_iter)
    if nb_pierre_pass-echec != 0:
        res = skfmm.distance(mask_array,dx=1)
        sort = np.argsort(-res,axis=None)
        index = np.unravel_index(sort[0:int(pas):int(pas/nb_pierre_pass)], res.shape)
        index = np.array(index).transpose()
        index  = index.tolist()
    
plt.savefig("../Resultat/placementPierre/courbes.png",dpi=150)

plt.figure()
plt.imshow(environnement,origin="lower")
plt.savefig("../Resultat/placementPierre/final.png",dpi=150)
plt.show()

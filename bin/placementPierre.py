import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,misc,sparse,stats
from math import inf,sqrt
import skfmm

### Importation des données
image = np.load("arraySave/image.npy")
mask = sparse.load_npz("arraySave/MaskChemin.npz")
environnement = cv2.imread("arraySave/environnement.jpg")
#environnement = cv2.cvtColor(environnement, cv2.COLOR_BGR2RGB)
ListeTaillePierre = np.load("arraySave/ListTaillePierre.npy" )
Pierre_mask = np.load("arraySave/PierreMask.npy",allow_pickle=True)
Pierre_sorted = np.load("arraySave/PierreSorted.npy",allow_pickle=True)

mask_array = -mask.toarray()+1
#res = ndimage.gaussian_filter(mask.toarray(),sigma =10,order=2)
## Fast marching
res = skfmm.distance(mask_array,dx=1)
[maskl,maskL] = mask_array.shape


### Algorithme de placement des pierre
k_max=0
T = 1
alpha = 0.9
num_pierre_max = len(Pierre_mask)
stop=False
k = 0
nb_try = 2
while not stop:
    echec_try = 0
    for trys in range(nb_try):
        ### Meilleur placement trouver par fast_marching
        sort = np.argsort(-res,axis=None)
        echec = 0
        pas = round(len(sort)/200)
        ### Placement de T pierre
        for i in range(10):
            index = np.unravel_index(sort[i*int(pas/10)], res.shape)
            ## Selection de la pierre
            num_pierre = stats.poisson.rvs(T)
            if (num_pierre >= num_pierre_max):
                num_pierre = num_pierre_max -1
            mask_pierre = Pierre_mask[num_pierre]
            pierre = Pierre_sorted[num_pierre]

            ## Dimension pierre
            [l_pierre,L_pierre] = mask_pierre.shape
            xmin = index[0]-int(l_pierre/2)
            xmax = index[0]+int(l_pierre/2+0.5)
            ymin = index[1]-int(L_pierre/2)
            ymax = index[1]+int(L_pierre/2+0.5)
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
            else:
                echec = echec+1
            # if k%10 == 0:
            #     fig = plt.figure() 
            #     ax1 = plt.subplot(1,2,1)
            #     ax1.imshow(mask_array,origin="lower")
            #     ax2 = plt.subplot(1,2,2)
            #     ax2.imshow(environnement,origin="lower")
            #     plt.show()
            k = k +1
        if echec > 5:
            echec_try = echec_try +1
    if (echec_try == nb_try):
        T = T + 1
    else:
        if(T > 1):
            T = T - 1
    if T > 13:
        stop = True
    print("T:")
    print(T)
    res = skfmm.distance(mask_array,dx=1)
    

plt.imshow(environnement,origin="lower")
plt.show()

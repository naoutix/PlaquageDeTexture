import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,misc,sparse
from math import inf,sqrt
import skfmm

### Importation des données
image = np.load("arraySave/image.npy")
mask = sparse.load_npz("arraySave/MaskChemin.npz")
environnement = img = cv2.imread("arraySave/environnement.jpg")
ListeTaillePierre = np.load("arraySave/ListTaillePierre.npy" )
Pierre_mask = np.load("arraySave/PierreMask.npy",allow_pickle=True)
Pierre_sorted = np.load("arraySave/PierreSorted.npy",allow_pickle=True)

mask_array = -mask.toarray()+1
#res = ndimage.gaussian_filter(mask.toarray(),sigma =10,order=2)

## Fast marching
res = skfmm.distance(mask_array,dx=1)
k=0
while k==2:
    index = res.argmax()
    k = k+1

plt.imshow(res,cmap="gray",origin="lower")
plt.show()

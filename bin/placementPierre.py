import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,misc,sparse


image = np.load("arraySave/image.npy")
mask = sparse.load_npz("arraySave/MaskChemin.npz")
environnement = img = cv2.imread("arraySave/environnement.jpg")
ListeTaillePierre = np.load("arraySave/ListTaillePierre.npy" )
Pierre_mask = np.load("arraySave/PierreMask.npy",allow_pickle=True)
Pierre_sorted = np.load("arraySave/PierreSorted.npy",allow_pickle=True)

res = ndimage.gaussian_filter(mask.toarray(),sigma =10,order=2)
res = res[3000:4000,1000:2000]
print(res.shape)
np.savetxt("res",res,fmt="%d")
fig = plt.figure()
plt.imshow(res,cmap="gray",origin="lower")
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,misc


image = np.load("arraySave/image.npy")
mask = np.load("arraySave/MaskChemin.npy")
environnement = np.load("arraySave/environnement.npy")
ListeTaillePierre = np.load("arraySave/ListTaillePierre.npy" )
Pierre_mask = np.load("arraySave/PierreMask.npy",allow_pickle=True)
Pierre_sorted = np.load("arraySave/PierreSorted.npy",allow_pickle=True)

res = ndimage.gaussian_filter(mask,sigma =10,order=2)
res = res[3000:4000,1000:2000]
print(res.shape)
np.savetxt("res",res,fmt="%d")
fig = plt.figure()
plt.imshow(res,cmap="gray",origin="lower")
plt.show()
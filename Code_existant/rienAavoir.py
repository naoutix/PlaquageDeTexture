import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,misc,sparse
from math import inf,sqrt
import skfmm
def fast_marching(frontiere):
    # 0 - calculé
    # 1 - proche
    # 2 - restant

    [l_max,L_max] = frontiere.shape
    lin_ind = np.argwhere(frontiere == 0)
    nb_point_frontiere = len(lin_ind)
    voisin=np.array([[1,0],[-1,0],[0,1],[0,-1]])
    # Distance infinie
    D=np.ones(frontiere.shape)*inf
    labels=2*np.ones(frontiere.shape)

    # Mise en place des points calcule
    D[tuple(np.transpose(lin_ind))]=0
    labels[tuple(np.transpose(lin_ind))]=0

    print("init")
    proche = []
    ## intialisation
    for i in range(nb_point_frontiere):
        for k in [0,1,-1]:
            for l in [0,1,-1]:
                x_k =lin_ind[i,0]+k
                y_k =lin_ind[i,1]+l
                # test des limite

                if (x_k >= 0 and x_k <l_max and y_k >=0 and y_k<L_max):
                    # test deja calculé
                    if labels[x_k,y_k] != 0:
                        labels[x_k,y_k] = 1;
                        proche.append([x_k,y_k])
    print("fin init")
    ## fast marching
    stop=False;
    while ~stop:
        longeur_proche = len(proche)
        print(longeur_proche)
        if longeur_proche != 0:
            index = np.argsort(D[tuple(np.transpose(np.array(proche)))])
            for pixel_proche in range(longeur_proche):
                #print(index[pixel_proche])
                p1 = proche[index[pixel_proche]]
                #print(p1)
                # passage a calculé
                labels[tuple(p1)]=0
                min_dist = inf
                for k in [0,1,-1]:
                    for l in [0,1,-1]:
                        x_k =p1[0]+k
                        y_k =p1[1]+l
                        # test des limite
                        if x_k >= 0 and x_k <l_max and y_k >=0 and y_k<L_max:
                            # test deja calculé
                            if labels[x_k,y_k] == 0:
                                d_voisin = sqrt(k*k+l*l) + D[x_k,y_k]
                                if d_voisin < min_dist:
                                    min_dist = d_voisin
                            else: 
                                if labels[x_k,y_k] == 2:
                                    labels[x_k,y_k]=1
                                    proche.append([x_k,y_k])
                D[tuple(p1)] = min_dist
            for i in range(longeur_proche):
                proche.pop(0)
        else:
            stop=True
    return D

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,sparse

def create_background(patch_background_elem,l,L):
    ###### Creation du background
    patch_colonne = np.copy(patch_background_elem)
    for i in range(L):
        patch_colonne = np.concatenate((patch_colonne,patch_background_elem),axis=0)

    environnement = np.copy(patch_colonne)
    for i in range(l):
        environnement = np.concatenate((environnement,patch_colonne),axis=1)
    return environnement

def selection_chemin(environnement,max_longeur_chemin):
    fig  = plt.figure(figsize=(13, 7), dpi=150)
    ax = plt.subplot(1,2,1)
    ax.set_title('Selection du chemin')
    ax.imshow(environnement,origin='lower')
    points = plt.ginput(max_longeur_chemin)   #Donnee Points
    return points,fig,ax

def interpolation(points,nbPoint):
    pointx = np.asarray(points)[:,0]
    pointy = np.asarray(points)[:,1]
    # Spline Parametrique (NbPoints > 4)
    Points_parametrisation = np.linspace(0, 1.01, nbPoint)
    tck, u = interpolate.splprep([pointx, pointy],k=2)
    chemin = interpolate.splev(Points_parametrisation, tck)
    return chemin

def largeur_R_chemin(Taille_chemin,environnement,chemin):
    nbPoint = len(chemin[0])
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
        
        # Verification limite
        xlimite_d = xc-Taille_chemin
        xlimite_g = xc+Taille_chemin+1
        ylimite_d = yc-Taille_chemin
        ylimite_g = yc+Taille_chemin+1
        xcercle_d = 0
        xcercle_g = 2*Taille_chemin+1
        ycercle_d = 0
        ycercle_g = 2*Taille_chemin+1
        if (xc-Taille_chemin < 0):
            xlimite_d = 0
            xcercle_d = Taille_chemin-xc
        if (xc+Taille_chemin+1 >= maskl):
            xlimite_g = maskl-1
            xcercle_g = 2*Taille_chemin - (xc+Taille_chemin+1 - maskl)
        if (yc-Taille_chemin < 0):
            ylimite_d = 0
            ycercle_d = Taille_chemin-yc
        if (yc+Taille_chemin+1 >= maskL):
            ylimite_g = maskL-1
            ycercle_g = 2*Taille_chemin - (yc+Taille_chemin+1 - maskL)
        mask[xlimite_d:xlimite_g,ylimite_d:ylimite_g]=mask[xlimite_d:xlimite_g,ylimite_d:ylimite_g]*cercle[xcercle_d:xcercle_g,ycercle_d:ycercle_g]     
    mask_bsr = sparse.bsr_matrix(mask)
    return mask_bsr

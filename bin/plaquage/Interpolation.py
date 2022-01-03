import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape 
from scipy import interpolate,ndimage,sparse,stats

def create_background(patch_background_elem,l,L):
    """ Creer le background de taille l*L a partir du un patch de texture  

    Args:
        patch_background_elem (numpy.ndarray): patch de texture
        l (int): taille en ligne
        L (int): taille en colonne

    Returns:
        numpy.ndarray: l'environnement creer
    """
    ###### Creation du background
    patch_colonne = np.copy(patch_background_elem)
    for i in range(L):
        patch_colonne = np.concatenate((patch_colonne,patch_background_elem),axis=0)

    environnement = np.copy(patch_colonne)
    for i in range(l):
        environnement = np.concatenate((environnement,patch_colonne),axis=1)
    return environnement

def selection_chemin(environnement,max_longeur_chemin=10):
    """Selection interactif de points 

    Args:
        environnement (numpy.ndarray): background
        max_longeur_chemin (int): taille maximal avant l'arret interactif

    Returns:
        List[Tuple[float,float]]: points-> points selectionner
        Figure                  : fig -> figure 
        SubplotBase             : ax -> ax de la figure 
    """
    fig  = plt.figure(figsize=(13, 7), dpi=150)
    ax = plt.subplot(1,2,1)
    ax.set_title('Selection du chemin')
    ax.imshow(environnement,origin='lower')
    points = plt.ginput(max_longeur_chemin)   #Donnee Points
    return points,fig,ax

def interpolation_Splines(points,nbPoint,k=2):
    """Interpolation des points selon une Spline

    Args:
        points (List[tuple[float,float]]): List des points
        nbPoint (int): nombre de points utilisé pour la spline
        k (int) : ordre de la spline

    Returns:
        List[tuple[float,float]]: Points de la spline interpolé
    """
    pointx = np.asarray(points)[:,0]
    pointy = np.asarray(points)[:,1]
    # Spline Parametrique (NbPoints > 4)
    Points_parametrisation = np.linspace(0, 1.01, nbPoint)
    tck, u = interpolate.splprep([pointx, pointy],k=k)
    chemin = interpolate.splev(Points_parametrisation, tck)
    return chemin

def limite_changer(Chemin_lim,Cercle_index,l,L):
    xlim_d      = Chemin_lim[0].start
    xcercle_d   = Cercle_index[0].start
    xlim_g      = Chemin_lim[0].stop
    xcercle_g   = Cercle_index[0].stop
    ylim_d      = Chemin_lim[1].start
    ycercle_d   = Cercle_index[1].start
    ylim_g      = Chemin_lim[1].stop
    ycercle_g   = Cercle_index[1].stop
    if (Chemin_lim[0].start < 0):
        xcercle_d = -Chemin_lim[0].start
        xlim_d = 0
    if (Chemin_lim[0].stop >= l):
        xcercle_g = Cercle_index[0].stop - (Chemin_lim[0].stop - (l-1))
        xlim_g = l-1
    if (Chemin_lim[1].start < 0):
        ycercle_d = -Chemin_lim[1].start
        ylim_d = 0
    if (Chemin_lim[1].stop >= L):
        ycercle_g = Cercle_index[1].stop - (Chemin_lim[1].stop - (L-1))
        ylim_g = L-1
    return tuple([slice(xlim_d,xlim_g,None),slice(ylim_d,ylim_g,None)]),tuple([slice(xcercle_d,xcercle_g,None),slice(ycercle_d,ycercle_g,None)])
    

def largeur_R_chemin(Taille_chemin,environnement,chemin):
    """Creation d'un masque en fonction d'un chemin dans un environnemment

    Args:
        Taille_chemin (int): Taille du chemin voulue en pixels
        environnement (numpy.ndarray): environnement :TODO -> juste mettre la taille
        chemin (List[tuple[float,float]]): Points du chemin

    Returns:
        bsr_matrix: Masque contenant des 1 la où il y a le chemin et 0 sinon
    """
    nbPoint = len(chemin[0])
    aleatoire = round(Taille_chemin)
    x=np.arange(-aleatoire,aleatoire+1,1)
    X,Y = np.meshgrid(x,x)
    # Forme autour du points à appliquer
    cercle = ~(X*X+Y*Y < aleatoire*aleatoire)
    cercle = cercle.astype(int)
    #Limite
    [maskl,maskL,_] = environnement.shape
    
    mask = np.ones((maskl,maskL))
    for i in range(nbPoint):

        yc = round(chemin[0][i])
        xc = round(chemin[1][i])
        
        # Verification limite
        Chemin_lim = [slice(xc-aleatoire,xc+aleatoire+1,None),slice(yc-aleatoire,yc+aleatoire+1,None)]
        Cercle_index = [slice(0,2*aleatoire+1),slice(0,2*aleatoire+1)]
        Chemin_lim,Cercle_index = limite_changer(Chemin_lim,Cercle_index,maskl,maskL)
        # Multiplication par le mask
        mask[Chemin_lim]=mask[Chemin_lim]*cercle[Cercle_index]     
    mask_bsr = sparse.bsr_matrix(mask)
    return mask_bsr

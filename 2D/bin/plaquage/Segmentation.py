import cv2
from matplotlib.colors import BoundaryNorm
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append 
from scipy import interpolate,ndimage
from math import sqrt

def segmentation(Texture,Reverse,Repetable):
    nbligne,nbcol,canaux = Texture.shape
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = Texture.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(Texture.shape)

    ###### Labelisation des pierre
    label_image = labels.reshape(nbligne,nbcol)
    image_valeur = Texture
    if Reverse:
        label_image = -labels.reshape(nbligne,nbcol)+1
    if Repetable:
        label_image = np.concatenate((label_image,label_image),axis=0)
        label_image = np.concatenate((label_image,label_image),axis=1)
        image_valeur = np.concatenate((Texture,Texture),axis=0)
        image_valeur = np.concatenate((image_valeur,image_valeur),axis=1)
    return label_image,image_valeur,segmented_image

def sep_box_pierre(label_image,image_valeur,Edge_cutting):
    nbligne,nbcol,canaux = image_valeur.shape
    # Separation des differentes pierres
    pierre_segmenter, num_pierre = ndimage.label(label_image)

    # BoundingBox
    boundingBox = ndimage.find_objects(pierre_segmenter)

    ### Triage des pierre selon leur taille

    # Triage des pierre par Taille en pixel + decoupage et mise dans une boundingBox
    ListTaillePierre = []
    Pierres = []
    Pierres_mask = []
    for pierre in range(num_pierre):
        boundingBoxPierre = (boundingBox[pierre][0],boundingBox[pierre][1])

        ## Taille de la pierre
        index_pierre = np.argwhere(pierre_segmenter[boundingBoxPierre] == pierre+1)
        taillePierre = len(index_pierre)
        
        ## Decoupage
        pierre_image = np.copy(image_valeur[boundingBoxPierre])
        pierre_masque = (pierre_segmenter[boundingBoxPierre] == pierre+1)
        for i in range(3):
            pierre_image[:,:,i] = (pierre_segmenter[boundingBoxPierre] == pierre+1)*pierre_image[:,:,i]
        
        flag = False
        ## Stockage
        if Edge_cutting:
            if (boundingBox[pierre][0].start == 0 or boundingBox[pierre][0].stop == nbligne or boundingBox[pierre][1].start == 0 or boundingBox[pierre][1].stop == nbcol):
                flag = True
        ## Suppresion doublon
        i = 0
        while i<len(Pierres) and not flag:
            if np.array_equal(pierre_image, Pierres[i] ):
                flag = True
            i = i +1
        if (not flag and taillePierre > 8):
            Pierres_mask.append(pierre_masque)
            Pierres.append(pierre_image)
            ListTaillePierre.append(taillePierre)
        num_pierre = len(Pierres)

    ## Trie
    Index_sort = np.argsort(-np.asarray(ListTaillePierre))
    Pierres_mask = np.take_along_axis(np.asarray(Pierres_mask),Index_sort , axis=0)
    Pierres = np.take_along_axis(np.asarray(Pierres),Index_sort , axis=0)
    ListTaillePierre = np.take_along_axis(np.asarray(ListTaillePierre), Index_sort, axis=0)
    return Pierres,Pierres_mask,ListTaillePierre,num_pierre,pierre_segmenter
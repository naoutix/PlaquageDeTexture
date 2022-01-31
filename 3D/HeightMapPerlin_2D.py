# Tutoriel: https://www.youtube.com/watch?v=aN05NVdiM8I

import cv2
import noise
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d


def set_color(x,y,image,value,limite):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]
        image ([type]): [description]
        value ([type]): [description]
    """    
    if value < limite:
        image.putpixel((x,y), blue)
    elif value < -0.1:
        image.putpixel((x,y), beach)
    elif value < 0.10:
        image.putpixel((x,y), green)
    elif value < 0.30:
        image.putpixel((x,y), mountains)
    else:
        image.putpixel((x,y), snow)

def set_texture(image,image_texture,couleur_texture):
    """[summary]

    Args:
        image ([type]): [description]
        image_texture ([type]): [description]
        couleur_texture ([type]): [description]

    Returns:
        [type]: [description]
    """    
    im = image.load()
    im_tex = image_texture.load()
    im_texture_finale = Image.new(mode="RGB",size=shape)
    # Application de la texture selon la couleur
    for x in range(shape[0]):
        for y in range(shape[1]):
            if (im[x,y] == couleur_texture):
                im_texture_finale.putpixel((x,y),im_tex[x,y])
            else:
                im_texture_finale.putpixel((x,y),im[x,y])
    return im_texture_finale

def paste_texture(file,nb,shape,environment,color):
    texture_elem = Image.open(file)
    bg_w, bg_h = texture_elem.size
    texture = Image.new('RGB', tuple(i*nb for i in shape ))
    
    w, h = texture.size
    for i in range(0, w, bg_w):
        for j in range(0, h, bg_h):
            texture.paste(texture_elem, (i, j))
    final = texture.resize((shape[0],shape[1]))
    return set_texture(environment,final,color)

## Constante
grandeur= 3
shape = (512*grandeur,512*grandeur)

scale = 100.0*grandeur
octaves = 8
lacunarity = 2.0
persistence = 0.5

# Couleur
blue = (66,110,225)
green = (36,135,32)
beach = (240,210,172)
mountains = (140,140,140)
snow = (250,250,250)

image_filepath = "noise.png"

image = Image.new(mode="RGB",size=shape)

values =np.zeros(shape)
limite = -0.2
for x in range(shape[0]):
    for y in range(shape[1]):
        value = noise.pnoise2(x/scale,
                            y/scale,
                            octaves=octaves,
                            persistence=persistence,
                            lacunarity=lacunarity,
                            repeatx=shape[0],
                            repeaty=shape[1],
                            base=0)
        set_color(x,y,image,value,limite)
        value = value-limite
        if value<0:
            value = 0
        values[y,x] = value
# max_arr = np.max(values)
# min_arr = np.min(values)
# norm_me = lambda x: (x)/(max_arr)
# norm_me = np.vectorize(norm_me)
# arr = norm_me(values)
arr = np.round(values*np.round(255/(1-limite)))
cv2.imwrite("perlin.png",arr)
image.save(image_filepath)

# Application de la texture sur chaque couleur correspondant à la texture
# Mer:
texture_finale = paste_texture("../2D/Textures/freeTexture4.png",grandeur*2,shape,image,blue)
# Herbe:
texture_finale = paste_texture("../2D/Textures/freeTexture2.png",grandeur*2,shape,texture_finale,green)
# Plage:
texture_finale = paste_texture("../2D/Textures/freeTexture1.png",grandeur*2,shape,texture_finale,beach)
# Montagne:
texture_finale = paste_texture("../2D/Textures/freeTexture13.png",grandeur*2,shape,texture_finale,mountains)
# Neige:
texture_finale = paste_texture("../2D/Textures/freeTexture15.png",grandeur*2,shape,texture_finale,snow)
plt.imshow(texture_finale)
plt.show()
texture_finale.save("texture_finale.png")

# Mettre la carte de hauteur en 3D ou coller la texture avec les chemins sur la texture 3D
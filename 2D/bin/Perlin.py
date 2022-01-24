# Tutoriel: https://www.youtube.com/watch?v=aN05NVdiM8I

import noise
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np

import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import axes3d

shape = (512,512)


#### Bruit de Perlin
scale = 50.0
octaves = 2
lacunarity = 2.0
persistence = 1.2


# Couleur RGB
blue = (66,110,225)
green = (36,135,32)
beach = (240,210,172)
mountains = (140,140,140)
snow = (250,250,250)

image_filepath = "noise.jpg"

image = Image.new(mode="RGB",size=shape)

def set_color(x,y,image,value):
    if value < -0.07:
        image.putpixel((x,y), blue)
    elif value < 0:
        image.putpixel((x,y), beach)
    elif value < 0.25:
        image.putpixel((x,y), green)
    elif value < 0.5:
        image.putpixel((x,y), mountains)
    elif value < 1.0:
        image.putpixel((x,y), snow)

def set_texture(image,image_texture,couleur_texture):
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

Perlin = np.zeros(shape)
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
        value = np.zeros()
        set_color(x,y,image,value)
print(value)
image.save(image_filepath)

# lin_x = np.linspace(0,1,shape[0],endpoint=False)
# lin_y = np.linspace(0,1,shape[1],endpoint=False)
# x,y = np.meshgrid(lin_x,lin_y)
# fig = matplotlib.pyplot.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_surface(x,y,value,facecolors=image)
# pyplot.show()

# Application de la texture sur chaque couleur correspondant à la texture
# Mer:
image_mer = Image.open("Textures/freeTexture4.jpg")
bg_w, bg_h = image_mer.size
mer = Image.new('RGB', (3000,3000))
w, h = mer.size
for i in range(0, w, bg_w):
    for j in range(0, h, bg_h):
        mer.paste(image_mer, (i, j))
mer = mer.resize((shape[0],shape[1]))
texture_finale = set_texture(image,mer,blue)
# Herbe:
image_herbe = Image.open("Textures/freeTexture2.jpg")
bg_w, bg_h = image_herbe.size
herbe = Image.new('RGB', (3000,3000))
w, h = herbe.size
for i in range(0, w, bg_w):
    for j in range(0, h, bg_h):
        herbe.paste(image_herbe, (i, j))
herbe = herbe.resize((shape[0],shape[1]))
texture_finale = set_texture(texture_finale,herbe,green)
# Plage:
image_mer = Image.open("Textures/freeTexture1.jpg")
bg_w, bg_h = image_mer.size
mer = Image.new('RGB', (3000,3000))
w, h = mer.size
for i in range(0, w, bg_w):
    for j in range(0, h, bg_h):
        mer.paste(image_mer, (i, j))
plage = mer.resize((shape[0],shape[1]))
texture_finale = set_texture(texture_finale,plage,beach)
# Montagne:
image_mer = Image.open("Textures/freeTexture13.jpg")
bg_w, bg_h = image_mer.size
mer = Image.new('RGB', (3000,3000))
w, h = mer.size
for i in range(0, w, bg_w):
    for j in range(0, h, bg_h):
        mer.paste(image_mer, (i, j))
montagne = mer.resize((shape[0],shape[1]))
texture_finale = set_texture(texture_finale,montagne,mountains)
# Neige:
image_mer = Image.open("Textures/freeTexture15.jpg")
bg_w, bg_h = image_mer.size
mer = Image.new('RGB', (3000,3000))
w, h = mer.size
for i in range(0, w, bg_w):
    for j in range(0, h, bg_h):
        mer.paste(image_mer, (i, j))
neige = mer.resize((shape[0],shape[1]))
texture_finale = set_texture(texture_finale,neige,snow)
texture_finale.save("texture_finale.jpg")

# Mettre la carte de hauteur en 3D ou coller la texture avec les chemins sur la texture 3D
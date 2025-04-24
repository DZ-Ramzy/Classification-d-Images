import math
import random

import numpy as np
import matplotlib.image as mpim
import cv2
from matplotlib import pyplot as plt

def rgb_vers_gris(img):
    """Retourne l'image en niveau de gris si elle ne l'est pas déjà."""

    if len(img.shape) != 2:
        im_gris = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                im_gris[i, j] = int(img[i, j, 0] * 1.0 + img[i, j, 1] * 1.0 + img[i, j, 2] * 1.0) // 3
        return im_gris
    else:
        return img

def euclide_dist(p1, p2):
    #print(f"p1 = {p1}, p2 = {p2}")
    return math.sqrt(math.pow(p1-p2, 2))

def enlever_fond(img, max_iter, k=2):
    """Enlève le fond de l'image à l'aide k-means, utile pour éviter de réaliser des opérations
    sur des parties inutiles de l'image, nottament pour la détection des lignes plus tard."""

    centroids = []

    #création des centroids (valeurs dans la diagonale de l'image, choix arbitraire)
    for x in range(k):
        #centroids.append(img[x,x])
        #print(type(np.max(img)))
        centroids.append(random.randint(int(np.min(img)), int(np.max(img))))

    for y in range(max_iter):

        classes = []

        for x in range(k):
            c = []
            classes.append(c)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dist = []
                for pos in centroids:
                    dist.append(euclide_dist(int(img[i,j]), int(pos)))
                index = dist.index(min(dist))
                classes[index].append(img[i,j])

        precedents = centroids.copy()
        for index in range(len(classes)):
            centroids[index] = np.mean(classes[index])

        opti = True
        for i in range(len(centroids)):
            if centroids[i] != precedents[i]:
                opti = False

        if opti:
            break

    img_sortie = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if abs(int(img[i, j]) - int(centroids[0])) > abs(int(img[i, j]) - int(centroids[1])):
                img_sortie[i, j] = 1

    #print(centroids)
    return img_sortie


def canny(img):
    """Retourne une image en noir et blanc avec uniquement les contours de visible."""

    return cv2.Canny(img, 50, 250)

def hough(img):
    """Retourne une image en noir et blanc avec uniquement les lignes de visible, utilisation de la transformée de Hough."""


img = "../images/49.jpg"
img_np = mpim.imread(img)

img_np_gris = rgb_vers_gris(img_np)
img_np_gris_sans_fond = enlever_fond(img_np_gris, 10, 2)

plt.imshow(img_np_gris_sans_fond, cmap='gray')
plt.show()

#img_finale = canny(img_np_gris_sans_fond)
img_finale = canny(img_np_gris)

plt.imshow(img_finale, cmap='gray')
plt.show()

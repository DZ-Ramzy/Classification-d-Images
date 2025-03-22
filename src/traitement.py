import numpy as np
import matplotlib.image as mpim
import statistics
from matplotlib import pyplot as plt
import random

def k_means(img, k):

    centroids = tuple((random.randrange(0,255), random.randrange(0,255)))

    x = 0
    while x < k:

        classe1_masque = abs(centroids[0] - img) < abs(centroids[1] - img)
        classe2_masque = abs(centroids[0] - img) >= abs(centroids[1] - img)

        classe1 = img[classe1_masque]
        classe2 = img[classe2_masque]

        if classe1.size > 0 and classe2.size > 0:
            nouv_centroid1 = statistics.mean(classe1)
            nouv_centroid2 = statistics.mean(classe2)

            if centroids[0] == nouv_centroid1 and centroids[1] == nouv_centroid2:
                break
            else:
                centroids = (nouv_centroid1, nouv_centroid2)

        x += 1

    return centroids

def seuillage(img, centroids):

    img2 = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if abs(img[i,j] - centroids[0]) > abs(img[i,j] - centroids[1]):
                img2[i,j] = 1

    return img2

def supprimer_fond(img_name, k):

    img_traitement = np.array(mpim.imread(img_name))
    if len(img_traitement.shape) != 2:
        img_traitement = np.dot(img_traitement[...,:3], [0.2989, 0.5870, 0.1140])

    centroids = k_means(img_traitement, k)
    img_seuil = seuillage(img_traitement, centroids)

    img = np.array(mpim.imread(img_name))

    masque = img_seuil == 0
    img[masque] = 0

    plt.imshow(img)
    plt.show()

supprimer_fond("3.jpg", 100)

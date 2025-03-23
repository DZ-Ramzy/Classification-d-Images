import numpy as np
import matplotlib.image as mpim
import statistics
from matplotlib import pyplot as plt

def variance(list):

    moy = statistics.mean(list)


def k_means(img_name, k):

    img = np.array(mpim.imread(img_name)).astype(int)
    if len(img.shape) != 2:
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    centroids = tuple((50,100))

    verif = True
    x = 0
    while x < k:

        """
                for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if abs(centroids[0] - img[i,j]) > abs(centroids[1] - img[i,j]):
                    classe2.append(img[i,j])
                else:
                    classe1.append(img[i,j])
                #print(img[i,j])
        """

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

def seuillage(img_name, centroids):

    img = np.array(mpim.imread(img_name))
    if len(img.shape) != 2:
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(int)

    img2 = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if abs(img[i,j] - centroids[0]) > abs(img[i,j] - centroids[1]):
                img2[i,j] = 1

    #plt.imshow(img2, interpolation='nearest')
    #plt.show()

    return img2

def supprimer_fond(img_name, k):

    #au lieu d'utiliser l'image utiliser direct np.array de l'image
    centroids = k_means(img_name, k)
    img_seuil = seuillage(img_name, centroids)

    img = np.array(mpim.imread(img_name))

    masque = img_seuil == 1
    img[masque] = 1

    plt.imshow(img)
    plt.show()


#print(seuillage("4.jpg",k_means("4.jpg", 5)))
supprimer_fond('C:/Users/alger/OneDrive/Desktop/projet_image/images/5.jpg', 69)
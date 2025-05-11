import math
import numpy as np
import cv2

def rgb_vers_gris(img):
    #Conversion de l'image en niveaux de gris, image en couleurs pas utile + les algos fonctionnent que sur image en niveaux de gris.
    
    if len(img.shape) != 2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img

def ouverture(img):
    #Erosion puis dilatation.
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12,4))
    
    erosion = cv2.erode(img, kernel)
    dilatation = cv2.dilate(erosion, kernel)
    
    return dilatation

def euclide_dist(p1, p2):
    #Distance entre un point et un centroid, pour savoir dans quelle classe mettre le point/
    
    return math.sqrt(math.pow(p1 - p2, 2))

def k_means(img, max_iter, k):
    #Seuillage avec kmeans, on veut supprimer le fond donc on prendra k = 2.
    
    flat_img = img.flatten().astype(np.float64)

    centroids = [flat_img[np.random.randint(len(flat_img))]]
    for _ in range(1, k):
        distances = np.array([min((x - c) ** 2 for c in centroids) for x in flat_img])
        probs = distances / distances.sum()
        cum_probs = np.cumsum(probs)
        r = np.random.rand()
        index = np.searchsorted(cum_probs, r)
        centroids.append(flat_img[index])
    centroids = np.array(centroids)

    for _ in range(max_iter):
        centroids_pre = centroids.copy()
        dist = np.abs(flat_img[:, np.newaxis] - centroids)
        indices = np.argmin(dist, axis=1)
        for i in range(k):
            points = flat_img[indices == i]
            centroids[i] = points.mean() if len(points) > 0 else centroids[i]
        if np.array_equal(centroids, centroids_pre):
            break

    return indices, centroids

def enlever_fond(img, max_iter, k):
    #Fonction pour enlever le fond avec kmeans, créé un masque et met un pixel blanc sur les pixels de la classe du fond.
    
    ind, centro = k_means(img, max_iter, k)

    centro = np.uint8(centro)
    img_seuil = centro[ind]
    img_seuil = img_seuil.reshape(img.shape)

    img_fond = np.argmax(centro)
    masque = (ind == img_fond).reshape(img.shape)
    img_result = img.copy()
    img_result[masque] = 255

    return img_result










































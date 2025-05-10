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
    return math.sqrt(math.pow(p1 - p2, 2))


def enlever_fond(img, max_iter, k=2):
    """Enlève le fond de l'image à l'aide de k-means."""
    centroids = [random.randint(int(np.min(img)), int(np.max(img))) for _ in range(k)]

    for _ in range(max_iter):
        classes = [[] for _ in range(k)]

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dist = [euclide_dist(int(img[i, j]), int(pos)) for pos in centroids]
                index = dist.index(min(dist))
                classes[index].append(img[i, j])

        precedents = centroids.copy()
        for index in range(len(classes)):
            if len(classes[index]) > 0:
                centroids[index] = np.mean(classes[index])

        if np.allclose(centroids, precedents):
            break

    img_sortie = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if abs(int(img[i, j]) - int(centroids[0])) > abs(int(img[i, j]) - int(centroids[1])):
                img_sortie[i, j] = 1

    return img_sortie


def canny(img):
    """Applique Canny sur une image."""
    return cv2.Canny(img, 50, 150)


def calculer_angle(ligne1, ligne2):
    """Calcule l'angle entre deux segments de ligne, ligne1 et ligne2."""
    x1, y1, x2, y2 = ligne1
    x3, y3, x4, y4 = ligne2

    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)

    # L'angle entre deux lignes
    angle_diff = abs(angle1 - angle2)
    return angle_diff


def fusionner_lignes_escalier(lignes, tolerance_position=10, tolerance_angle=0.1):
    """Fusionne les lignes adjacentes en une seule ligne par escalier."""
    fusionnees = []

    # Tri des lignes par leur coordonnée x de départ
    lignes = sorted(lignes, key=lambda l: l[0][0])

    # Ajouter la première ligne à la liste
    fusionnees.append([lignes[0][0][0], lignes[0][0][1], lignes[0][0][2], lignes[0][0][3]])

    for i in range(1, len(lignes)):
        x1, y1, x2, y2 = lignes[i][0]
        prev_x1, prev_y1, prev_x2, prev_y2 = fusionnees[-1]

        # Vérification de la proximité entre les lignes (position)
        if abs(x1 - prev_x2) < tolerance_position and abs(y1 - prev_y2) < tolerance_position:
            # Vérification de l'angle entre les lignes
            angle_diff = calculer_angle(lignes[i - 1][0], lignes[i][0])
            if angle_diff < tolerance_angle:
                # Fusionner les lignes en une seule
                fusionnees[-1] = [min(prev_x1, x1), min(prev_y1, y1), max(prev_x2, x2), max(prev_y2, y2)]
            else:
                # Sinon, ajouter la ligne à la liste
                fusionnees.append([x1, y1, x2, y2])
        else:
            # Ajouter la ligne à la liste si elle n'est pas proche de la précédente
            fusionnees.append([x1, y1, x2, y2])

    return fusionnees


def hough_lignes(img):
    """Retourne une image avec les segments détectés avec Hough probabiliste, sans lignes verticales et lignes fusionnées en escalier."""

    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    rho = 1
    theta = np.pi / 180
    threshold = 60
    min_line_length = 50
    max_line_gap = 5

    line_image = np.zeros((*img.shape, 3), dtype=np.uint8)  # Image couleur noire
    lignes_dessinees = []  # Liste pour stocker les lignes dessinées (non-verticales)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calcul de l'angle de la ligne
            angle = np.arctan2(y2 - y1, x2 - x1)
            # Filtrage des lignes verticales (angles proches de ±π/2)
            if not (np.abs(angle) > np.pi / 4):  # Lignes verticales (de -π/2 à π/2)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                lignes_dessinees.append(line)  # Ajouter la ligne à la liste des lignes dessinées

    # Fusionner les lignes adjacentes en escalier
    lignes_fusionnees = fusionner_lignes_escalier(lignes_dessinees)

    # Dessiner les lignes fusionnées en escalier
    for ligne in lignes_fusionnees:
        x1, y1, x2, y2 = ligne
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Lignes fusionnées en rouge

    return line_image, lignes_fusionnees


# --------- UTILISATION DU CODE ---------
img_path = r"C:\Users\alger\OneDrive\Desktop\projet_image\images\49.jpg"

img_np = mpim.imread(img_path)
img_np_gris = rgb_vers_gris(img_np)
img_np_gris_sans_fond = enlever_fond(img_np_gris, 10, 2)

# Affiche l'image sans fond
"""plt.figure(figsize=(10, 5))
plt.title("Image sans fond")
plt.imshow(img_np_gris_sans_fond, cmap='gray')
plt.axis('off')
plt.show()"""

# Applique Canny
img_canny = canny(img_np_gris)
plt.figure(figsize=(10, 5))
plt.title("Contours (Canny)")
plt.imshow(img_canny, cmap='gray')
plt.axis('off')
plt.show()

# Applique Hough
img_hough, lignes = hough_lignes(img_np_gris)
# print("Nombre de lignes détectées :", lignes)
plt.figure(figsize=(10, 5))
plt.title("Détection des lignes (Hough)")
plt.imshow(img_hough, cmap='gray')
plt.axis('off')
plt.show()
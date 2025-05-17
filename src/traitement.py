import statistics

import numpy as np
import cv2
import pretraitement as pt
import matplotlib.pyplot as plt


def fusionner_lignes_par_bande(lignes, hauteur_img, diviseur=20):
    #Méthode pour fusionner les lignes, un peu arbitraire, on divise la hauteur de l'image par un diviseur (hauteur parce qu'on regarde que les lignes horizontales), 
    #cette valeur nous servira pour créer des groupes de lignes qu'on fusionnera, pour un diviseur X et une ligne B si il y a des lignes entre B+X/2 et B-X/2 on les fusionne.
    
    if not lignes:
        return []

    bande_hauteur = hauteur_img // diviseur
    bandes = []

    for ligne in lignes:
        x1, y1, x2, y2 = ligne
        assignee = False
        for groupe in bandes:
            for gx1, gy1, gx2, gy2 in groupe:
                if (
                    abs(y1 - gy1) <= bande_hauteur and
                    abs(y2 - gy2) <= bande_hauteur
                ):
                    groupe.append(ligne)
                    assignee = True
                    break
            if assignee:
                break
        if not assignee:
            bandes.append([ligne])

    lignes_fusionnees = []
    for groupe in bandes:
        xs = [x1 for x1, _, x2, _ in groupe] + [x2 for x1, _, x2, _ in groupe]
        ys = [y1 for _, y1, _, y2 in groupe] + [y2 for _, y1, _, y2 in groupe]
        x1_f = min(xs)
        x2_f = max(xs)
        y_f = int(np.mean(ys))
        lignes_fusionnees.append((x1_f, y_f, x2_f, y_f))

    return lignes_fusionnees

def nettoyer_composantes(img):
    #On part du principe qu'on a une image binaire, on va se servir de la bibliothèque OpenCV

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    for i in range(1, num_labels):
        largeur = stats[i, cv2.CC_STAT_WIDTH]
        hauteur = stats[i, cv2.CC_STAT_HEIGHT]
        if largeur <= 300 or hauteur >= 600:
            img[labels == i] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    if num_labels >= 35:
        lar = []
        haut = []
        for i in range(1,num_labels):
            lar.append(stats[i, cv2.CC_STAT_WIDTH])
            haut.append(stats[i, cv2.CC_STAT_HEIGHT])
        moy_larg = statistics.mean(lar)
        moy_haut = statistics.mean(haut)
        img[stats[labels, cv2.CC_STAT_WIDTH] < moy_larg] = 0
        img[stats[labels, cv2.CC_STAT_HEIGHT] > moy_haut] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)

    print(f'connexes : {num_labels}')

    return num_labels, img

def calculer_composantes_connxes(img):


    #edges = cv2.Canny(img, 100, 200)
    edges_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5) #gradient sur x
    edges = cv2.convertScaleAbs(edges_x)
    #plt.imshow(edges)
    #plt.show()

    ouvert = pt.ouverture(edges)
    ferme = pt.fermeture(ouvert)

    #plt.imshow(ferme)
    #plt.show()

    #plt.imshow(ouvert)
    #plt.show()

    #nb_compo, ouvert = nettoyer_composantes(erosion)
    nb_compo, ouvert = nettoyer_composantes(ferme)



    return nb_compo

def hough_lignes(img):
    #Transformation de hough pour détecter les lignes de l'image, on prend que les lignes horizontales, chois arbitraire mais explicable
    #par notre base d'image. Ensuite on fusionne les lignes proches parce que sinon on a juste un gros amat de lignes.
    
    #ouvert = pt.ouverture(img)

    flou = cv2.GaussianBlur(img, (5,5), 1.5)
    edges = cv2.Canny(flou, 100, 200)
    ferme = pt.fermeture(edges)
    #ouvert = pt.ouverture(ferme)
    erosion = cv2.erode(ferme, cv2.getStructuringElement(cv2.MORPH_RECT, (1,5)))


    ouvert = nettoyer_composantes(erosion)

    #edges_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5) #gradient sur x
    #edges = cv2.convertScaleAbs(edges_x)


    lines = cv2.HoughLines(ouvert, 1, np.pi / 90, 150)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    lignes = []

    for line in lines:
        if line is not None:
            rho,theta = line[0]
            deg = np.rad2deg(theta) #on convertit theta en degrés avant
            
            if 95 >= deg >= 85:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                lignes.append((x1,y1,x2,y2))
                #cv2.line(color_img,(x1,y1),(x2,y2),(0,0,255),2)
                
    #lignes_fusionnees = fusionner_lignes_par_bande(lignes, img.shape[0], diviseur=20)

    for ligne in lignes:
        x1,y1,x2,y2 = ligne
        cv2.line(color_img,(x1,y1),(x2,y2),(0,0,255),4)

    plt.imshow(color_img)
    plt.show()

    
    #return color_img, lignes_fusionnees
    return lignes


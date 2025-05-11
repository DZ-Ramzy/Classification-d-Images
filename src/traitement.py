import numpy as np
import cv2
import pretraitement as pt

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
                    assignée = True
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


def hough_lignes(img, tolerance=20):
    #Transformation de hough pour détecter les lignes de l'image, on prend que les lignes horizontales, chois arbitraire mais explicable
    #par notre base d'image. Ensuite on fusionne les lignes proches parce que sinon on a juste un gros amat de lignes.
    

    ouvert = pt.ouverture(img)

    edges = cv2.Canny(ouvert, 50, 150, apertureSize=5)
    #edges_x = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5) #gradient sur x
    #edges = cv2.convertScaleAbs(edges_x)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
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
                
    lignes_fusionnees = fusionner_lignes_par_bande(lignes, img.shape[0], diviseur=30)
    
    for ligne in lignes_fusionnees:
        x1,y1,x2,y2 = ligne
        cv2.line(color_img,(x1,y1),(x2,y2),(0,0,255),4)
    
    return color_img, lignes_fusionnees


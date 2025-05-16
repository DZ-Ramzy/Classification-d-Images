import os

import pretraitement as pt
import traitement as tt
import matplotlib.image as mpim
import math
import csv

def calculer_lignes_base(chemin_base):
    #Permet de calculer le nombre de lignes pour chaque image de la base de données.

    images = os.listdir(chemin_base)

    nb_lignes = []
    
    for image in images:

        img = mpim.imread(chemin_base+"/"+image)
        img_gris = pt.rgb_vers_gris(img)
        img_sans_fond = pt.enlever_fond(img_gris, max_iter=20, k=2)
        _, lignes = tt.hough_lignes(img_sans_fond)
        nb_lignes.append((image.split('.')[0], len(lignes)))
    
    return nb_lignes

def mae(lignes, verite):
    #Calcule l'erreur absolue moyenne.
    
    #On part du principe qu'une marche est représentée par deux lignes, 
    #une ligne pour le début de la contremarche, et une pour le nez de la marche.
    marches = [x[1]//2 for x in lignes]
    
    res= 0
    for i in range(len(lignes)):
        res += abs(marches[i] - verite[i])
    
    return res/len(lignes)
    
    
def mse(lignes, verite):
    #Calcul de l'erreur quadraique moyenne.
    
    #On part du principe qu'une marche est représentée par deux lignes, 
    #une ligne pour le début de la contremarche, et une pour le nez de la marche.
    marches = [x[1]//2 for x in lignes]
    
    res = 0
    res= 0
    for i in range(len(lignes)):
        res += math.pow(marches[i] - verite[i], 2)
        
    return res/len(lignes)

def verite_en_f_de_base(fichier_verite, nb_lignes):
    #fonction pour récupérer uniquement les valeurs des images de la base consultée

    num_images = [x[0] for x in nb_lignes]

    fichier_verite = open(fichier_verite, mode='r')
    csv_verite = csv.reader(fichier_verite)
    verite = []
    for ligne in csv_verite:
        if ligne[1] != '' and ligne[0] in num_images:
            verite.append(int(ligne[1]))

    return verite

def base_test():

    chemin = "../test"
    nombre_lignes = calculer_lignes_base(chemin)
    veritee_coupee = verite_en_f_de_base('../verite/labels.csv', nombre_lignes)
    val_mae = mae(nombre_lignes, veritee_coupee)
    val_mse = mse(nombre_lignes, veritee_coupee)

    print(f'Valeur de la mae : {val_mae}.')
    print(f'Valeur de la mse : {val_mse}.')

def base_validation():

    chemin = "../validation"
    nombre_lignes = calculer_lignes_base(chemin)
    veritee_coupee = verite_en_f_de_base('../verite/labels.csv', nombre_lignes)
    val_mae = mae(nombre_lignes, veritee_coupee)
    val_mse = mse(nombre_lignes, veritee_coupee)

    print(f'Valeur de la mae : {val_mae}.')
    print(f'Valeur de la mse : {val_mse}.')
    
    
    
    
    
    
    
    
    
    
    
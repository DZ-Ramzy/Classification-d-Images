import os

import pretraitement as pt
import traitement as tt
import matplotlib.image as mpim
import math
import csv

def calculer_lignes_base(chemin_base):
    #Permet de calculer le nombre de lignes pour chaque image de la base de données.

    images = os.listdir(chemin_base)
    print(f'Images = {images}.')
    nb_lignes = dict()
    
    for image in images:
        img = mpim.imread(chemin_base+"/"+image)
        img_gris = pt.rgb_vers_gris(img)
        img_sans_fond = pt.enlever_fond(img_gris, max_iter=20, k=2)
        lignes = tt.hough_lignes(img_sans_fond)
        nb_lignes[int(image.split('.')[0])] = len(lignes)
    print(f'nb lignes = {nb_lignes}')
    return nb_lignes

def mae(lignes, verite):
    #Calcule l'erreur absolue moyenne.
    
    #On part du principe qu'une marche est représentée par deux lignes, 
    #une ligne pour le début de la contremarche, et une pour le nez de la marche.
    
    res= 0
    for key in lignes.keys():
        res += abs(lignes[key]//2 - verite[key])
    
    return res/len(lignes)
    
    
def mse(lignes, verite):
    #Calcul de l'erreur quadraique moyenne.
    
    #On part du principe qu'une marche est représentée par deux lignes, 
    #une ligne pour le début de la contremarche, et une pour le nez de la marche.

    res= 0
    for key in lignes.keys():
        print(f'clef, val, veri : {key}, {lignes[key]}, {verite[key]}')
        res += math.pow(lignes[key]//2 - verite[key], 2)
        
    return res/len(lignes)

def verite_en_f_de_base(fichier_verite, nb_lignes):
    #fonction pour récupérer uniquement les valeurs des images de la base consultée

    num_images = nb_lignes.keys()
    print(num_images)
    fichier_verite = open(fichier_verite, mode='r')
    csv_verite = csv.reader(fichier_verite)
    verite = dict()
    for ligne in csv_verite:
        if ligne[1] != '' and int(ligne[0]) in num_images:
            verite[int(ligne[0])] = int(ligne[1])

    return verite

def base_test():

    chemin = "../test"
    nombre_lignes = calculer_lignes_base(chemin)
    print(f'lignes = {nombre_lignes}')
    veritee_coupee = verite_en_f_de_base('../verite/labels.csv', nombre_lignes)
    print(f'veritee = {veritee_coupee}')
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
    
    
    
    
    
    
    
    
    
    
    
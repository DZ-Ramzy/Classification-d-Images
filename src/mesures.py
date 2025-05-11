import pretraitement as pt
import traitement as tt
import matplotlib.image as mpim
import math
import csv

def calculer_lignes_base(nb_img):
    #Permet de calculer le nombre de lignes pour chaque image de la base de données.
    
    nb_lignes = []
    
    for x in range(nb_img):
        chemin = "../images/"+str(x)+".jpg"
        
        try:
            img = mpim.imread(chemin)
            img_gris = pt.rgb_vers_gris(img)
            img_sans_fond = pt.enlever_fond(img_gris, max_iter=20, k=2)
            _, lignes = tt.hough_lignes(img_sans_fond)
            nb_lignes.append((x, len(lignes)))
            
        except Exception as E:
            print(f'le fichier {x}.jpg n\'existe pas')
            continue
    
    return nb_lignes

def mae(lignes, verite):
    #Calcule l'erreur absolue moyenne.
    
    #On part du principe qu'une marche est représentée par deux lignes, 
    #une ligne pour le début de la contremarche, et une pour le nez de la marche.
    marches = [x[1]//2 for x in lignes]
    
    res= 0
    for i in range(len(lignes)):
        res += abs(marches[i] - verite[i][1])
    
    return res/len(lignes)
    
    
def mse(lignes, verite):
    #Calcul de l'erreur quadraique moyenne.
    
    #On part du principe qu'une marche est représentée par deux lignes, 
    #une ligne pour le début de la contremarche, et une pour le nez de la marche.
    marches = [x[1]//2 for x in lignes]
    
    res = 0
    res= 0
    for i in range(len(lignes)):
        res += math.pow(marches[i] - verite[i][1], 2)
        
    return res/len(lignes)

def test(nb_img):
    
    fichier_verite = open('../verite/labels.csv', mode='r')
    csv_verite = csv.reader(fichier_verite)
    verite = []
    for ligne in csv_verite:
        if ligne[1] != '':
            verite.append((int(ligne[0]), int(ligne[1])))
        
    lignes = calculer_lignes_base(nb_img)
    
    mae_val = mae(lignes, verite)
    
    print(mae_val)
    
    
    
    
    
    
    
    
    
    
    
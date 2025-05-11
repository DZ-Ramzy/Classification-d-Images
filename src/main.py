import pretraitement as pt
import matplotlib.image as mpim
from matplotlib import pyplot as plt

if __name__ == '__main__':

    fichier = '../images/49.jpg'
    img = mpim.imread(fichier)
    img_gris = pt.rgb_vers_gris(img)
    img_sans_fond = pt.enlever_fond(img_gris, 10, 2)

    plt.figure(figsize=(10,5))
    plt.title('Test')
    plt.imshow(img_sans_fond, cmap='gray')
    plt.show()

    img_lignes = pt.hough_lignes(img_sans_fond)

    plt.figure(figsize=(20, 10))
    plt.title('Test 2')
    plt.imshow(img_lignes)
    plt.show()

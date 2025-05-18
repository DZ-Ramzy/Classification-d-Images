import mesures as ms
import pretraitement as pt
import traitement as tt
import matplotlib.image as mpim
import matplotlib.pyplot as plt
import cv2

def tester_chaine_sur_une_image(img):

    plt.figure(figsize=(12, 4))

    img = mpim.imread(img)
    img_gris = pt.rgb_vers_gris(img)
    img_norm = pt.egaliser_histogramme(img_gris)

    plt.subplot(2, 3, 1)
    plt.imshow(img_norm, cmap='gray')
    plt.title('Image après passage en niveau de gris et nomarlisation')
    plt.axis('off')

    #flou = cv2.GaussianBlur(img_norm, (5, 5), 5)
    flou = cv2.blur(img_norm, (15,15))

    plt.subplot(2, 3, 2)
    plt.imshow(flou, cmap='gray')
    plt.title('Image après flou gloussien, sigma = 5')
    plt.axis('off')

    img_sans_fond = pt.enlever_fond(flou, max_iter=20, k=2)
    compos, img_finale, ouvert = tt.calculer_composantes_connxes(img_sans_fond)

    #compos, img_finale, ouvert = tt.calculer_composantes_connxes(img_sans_fond)

    plt.subplot(2, 3, 3)
    plt.imshow(img_finale, cmap='gray')
    plt.title('Image sans fond et composantes nettoyées')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(ouvert, cmap='gray')
    plt.title('Ouvert')
    plt.axis('off')

    #plt.subplot(2, 3, 5)
    #plt.imshow(ferme, cmap='gray')
    #plt.title('Fermé de l\'ouvert')
    #plt.axis('off')

    print(f'Nombre de composantes connexes détectées après traitement : {compos}.')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    #tester_chaine_sur_une_image('../validation/22.jpg')
    ms.base_validation()
    #print()
    #ms.base_test()


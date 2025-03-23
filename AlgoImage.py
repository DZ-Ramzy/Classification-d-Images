from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpim

# Afficher une image passée en paramètre
def imshow(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Création d'une image noire, avec les paramètres width & height
def zeros(width, height):
    return Image.new('RGB', (width, height), (0, 0, 0))

# Convertir une image en image grise
def to_gray(img):
    return ImageOps.grayscale(img)

# Calculer le niveau de seuil d'une image, Seuillage OTSU
def otsu_threshold(image):
    histogram = image.histogram()
    total = image.size[0] * image.size[1]
    sum_total = sum(i * histogram[i] for i in range(256))

    sumB, wB, wF = 0, 0, 0
    varMax, threshold = 0, 0

    for i in range(256):
        wB += histogram[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * histogram[i]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        varBetween = wB * wF * (mB - mF) ** 2
        if varBetween > varMax:
            varMax = varBetween
            threshold = i

    return threshold

# Retourne une image seuillée à l'aide d'un niveau de seuil passé en paramètre
def threshold(img, s):
    img_array = np.array(img)
    thresholded_img = np.where(img_array < s, 0, 255)
    return Image.fromarray(thresholded_img.astype(np.uint8))

# Applique une convolution sur l'image, à l'aide d'un masque passé en paramètre
def convolve_op(img, kernel):
    return img.filter(ImageFilter.Kernel(kernel.size, kernel.flatten(), scale=kernel.sum()))

# Convertir une image en un tableau
def image_histogram(img):
    histogram = img.histogram()
    return histogram[:256]

# Retourne un tableau en pourcentage
def pourcentage(tab):
    max_val = max(tab)
    return [(x * 100) // max_val for x in tab]

# Retourne une image d'histogramme à l'aide du tableau
def ecrit_histo(tab):
    width = len(tab)
    height = 100
    img = zeros(width, height)
    for i in range(width):
        for j in range(height - tab[i], height):
            img.putpixel((i, j), (255, 0, 0))
    return img

# Retourne une image de l'histogramme de Niveau de Gris
def histo_g(img):
    tab = image_histogram(img)
    tab = pourcentage(tab)
    return ecrit_histo(tab)

# Trouver les deux coupures pour recadrer l'image
def coupure(tab):
    res = [0, 0]
    for i in range(len(tab)):
        if tab[i] > 10:
            res[0] = i
            break
    for i in range(res[0], len(tab)):
        if tab[i] < 10:
            res[1] = i
            break
    return res

# Trouver les deux coupures pour recadrer l'image (si la première méthode échoue)
def coupure2(tab):
    res = [0, 0]
    x = 0
    for i in range(len(tab)):
        if tab[i] > 75:
            x = i - 1
        if tab[i] == 100:
            res[0] = i - x
            m = i
            break
    for j in range(m, len(tab)):
        if tab[j] < 75:
            res[1] = j - 1
            break
    return res

# Retourne une image de l'histogramme de Projection, de couleur noire
def histo_r(img):
    width, height = img.size
    tab = [0] * width
    for x in range(width):
        for y in range(height):
            p = img.getpixel((x, y))
            if p == 0:
                tab[x] += 1
    tab = pourcentage(tab)
    res = coupure(tab)
    if res[0] == 0 and res[1] == 0:
        res = coupure2(tab)
    return ecrit_histo(tab)

# Retourne une nouvelle image en fonction de l'histogramme de Projection
def resize_image(img):
    width, height = img.size
    tab = [0] * width
    for x in range(width):
        for y in range(height):
            p = img.getpixel((x, y))
            if p == 0:
                tab[x] += 1
    tab = pourcentage(tab)
    res = coupure(tab)
    if res[0] == 0 and res[1] == 0:
        res = coupure2(tab)
    new_img = img.crop((res[0], 0, res[1], height))
    return new_img

# Retourne une nouvelle image coupée au centre de l'image
def coupure_centre(img):
    width, height = img.size
    new_img = img.crop((width // 3, 0, (width * 2) // 3, height))
    return new_img

# Retourne une nouvelle image coupée à droite de l'image
def coupure_droite(img):
    width, height = img.size
    new_img = img.crop((width * 2 // 3, 0, width, height))
    return new_img

# Retourne une nouvelle image coupée à gauche de l'image
def coupure_gauche(img):
    width, height = img.size
    new_img = img.crop((0, 0, width // 3, height))
    return new_img

def main():
    # Charger une image
    image_path = "C:\\Users\\alger\\OneDrive\\Desktop\\projet_image\\images\\5.jpg"
    #img = Image.open(image_path)

    img = np.array(mpim.imread(image_path))
    plt.imshow(img.k_means(img), cmap='gray', interpolation='nearest')
    plt.show()

    # Afficher l'image originale
    print("Image originale :")
    imshow(img)

    # Convertir l'image en niveaux de gris
    gray_img = to_gray(img)
    print("Image en niveaux de gris :")
    imshow(gray_img)

    # Calculer le seuil Otsu
    threshold_value = otsu_threshold(gray_img)
    print(f"Seuil Otsu : {threshold_value}")

    # Appliquer le seuillage
    thresholded_img = threshold(gray_img, threshold_value)
    print("Image seuillée :")
    imshow(thresholded_img)


    #imshow(hist_img)

if __name__ == "__main__":
    main()

# Projet UE Image

## Description
Ce projet vise à compter les marches sur des images d'escaliers, en python.

## Prérequis
Avant de commencer, assurez-vous d'avoir installé les bibliothèques suivantes :

- matplotlib
- numpy
- opencv

Vous pouvez les installer en utilisant pip :

```bash
pip install matplotlib numpy opencv-python
```

## Structure du projet

- `src/` : Contient tout le code source du projet.
  - `main.py` : Le fichier principal pour lancer le projet.
  - `mesures.py` : Contient les fonctions pour mesurer la précision de l'algorithme.
  - `pretraitement.py` : Contient les opérations de prétraitement effectuées avant la détection des escaliers.
  - `traitement.py` : Contient les fonctions pour détecter les marches dans les images.

- `test/` : Contient la base de test.
- `validation/` : Contient la base de validation.

## Utilisation

### Lancer le projet
Pour lancer le projet, exécutez le fichier `main.py` :

```bash
python main.py
```

### Tester la base de validation
Pour tester la base de validation, appelez la méthode `base_validation()` dans votre code.

### Tester la base de test
Pour tester la base de test, appelez la méthode `base_test()` dans votre code.

### Tester sur une seule image
Pour tester sur une seule image, appelez la fonction `tester_chaine_sur_une_image()` avec le chemin vers l'image en paramètre :

```python
tester_chaine_sur_une_image('chemin/vers/votre/image.jpg')
```

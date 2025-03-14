import cv2
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

# Paramètres de conversion
pixel_to_mm2 = 75.4 / 22959  # Conversion de pixels² en mm²
diameter_ref_px = 2 * np.sqrt(22959 / np.pi)  # Diamètre de la sphère de référence en pixels
diameter_ref_mm = 9.8  # Diamètre de la sphère en mm
pixel_to_mm = diameter_ref_mm / diameter_ref_px  # Conversion diamètre px → mm
temps = 0 #temps en secondes

# Dossier contenant les images
image_folder = "Methane_images"  # Remplace par le chemin de ton dossier
image_files = [f for f in os.listdir(image_folder)]

# Fichier CSV de sortie
csv_filename = "bulle_dissolution.csv"
file_exists = os.path.isfile(csv_filename)
#supprimerle fichier s'il existe
if os.path.exists(csv_filename):
    os.remove(csv_filename)

# Créer CSV
with open(csv_filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Aire (px²)", "Aire (mm²)", "Diamètre (px)", "Diamètre (mm)", "Temps (sec)"])

# Traitement des images
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)  # Charger l'image

    # Convertir en HSV
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Définir les seuils pour détecter la goutte
    bound_lower = np.array([0, 0, 0])
    bound_upper = np.array([20, 20, 20])
    mask_black = cv2.inRange(hsv_img, bound_lower, bound_upper)

    # Améliorer le masque avec des opérations morphologiques
    kernel = np.ones((7, 7), np.uint8)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)

    # Trouver les contours
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Prendre le plus grand contour (supposé être la bulle)
        c = max(contours, key=cv2.contourArea)

        # Calculer l'aire et le diamètre
        area_px = cv2.contourArea(c)
        area_mm2 = area_px * pixel_to_mm2
        diameter_px = 2 * np.sqrt(area_px / np.pi)
        diameter_mm = diameter_px * pixel_to_mm
        temps += 0.04

        # Ajouter au CSV
        with open(csv_filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([image_file, area_px, area_mm2, diameter_px, diameter_mm, temps])

        # Afficher la bulle détectée
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
    
    # 🖼️ Affichage
    cv2.imshow("Bulle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quitter avec 'q'
        break

with open(csv_filename, mode="r", newline="") as file:
    temps = []
    diametre = []
    lecture_csv = csv.DictReader(file)
    for ligne in lecture_csv:
        temps.append(float(ligne["Temps (sec)"]))
        diametre.append(float(ligne["Diamètre (mm)"]))

"""
print(temps)
print(diametre)
"""

plt.plot(temps, diametre)
plt.xlabel('Temps (sec)')
plt.ylabel('Diamètre (mm)')
plt.title("Évolution du diamètre de la goutte dans le temps")
plt.ylim(0, 2)
plt.show()


# Nettoyage
cv2.destroyAllWindows()

print(f"Résultats enregistrés dans {csv_filename}")

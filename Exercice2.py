import cv2
import numpy as np
import csv
import os

# Paramètres de conversion
pixel_to_mm2 = 75.4 / 22959  # Conversion de pixels² en mm²
diameter_ref_px = 2 * np.sqrt(22959 / np.pi)  # Diamètre de la sphère de référence en pixels
diameter_ref_mm = 9.8  # Diamètre de la sphère en mm
pixel_to_mm = diameter_ref_mm / diameter_ref_px  # Conversion diamètre px → mm
temps = 0 #temps en secondes

# Dossier contenant les images
image_folder = "Methane_images"  # Remplace par le chemin de ton dossier
image_files = [f for f in os.listdir(image_folder) ]

# Fichier CSV de sortie
csv_filename = "bulle_dissolution.csv"
file_exists = os.path.isfile(csv_filename)

# Créer CSV
with open(csv_filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Image", "Aire (px²)", "Aire (mm²)", "Diamètre (px)", "Diamètre (mm)", "Temps (sec)"])

# Traitement des images
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)  # Charger l'image

    # Convertir en niveaux de gris et filtrer
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1,1), 0)

    # Détection des contours
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Prendre le plus grand contour (supposé être la bulle)
        c = min(contours, key=cv2.contourArea)

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
    
    # Affichage
    cv2.imshow("Bulle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quitter avec 'q'
        break

# Nettoyage
cv2.destroyAllWindows()

print(f"Résultats enregistrés dans {csv_filename}")

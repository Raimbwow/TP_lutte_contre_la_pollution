import cv2
import numpy as np
import csv
import os

# Nom du fichier CSV pour enregistrer les résultats
csv_filename = "resultats_gouttes_mm.csv"

# Vérifier si le fichier existe déjà
file_exists = os.path.isfile(csv_filename)

# Facteurs de conversion
pixel_to_mm2 = 75.4 / 31000  # 1 pixel² ≈ 0.00243 mm²
diameter_ref_px = 2 * np.sqrt(31000 / np.pi)  # ≈ 198.4 pixels
diameter_ref_mm = 9.8
pixel_to_mm = diameter_ref_mm / diameter_ref_px  # 1 pixel ≈ 0.0494 mm

# Ouvrir le fichier CSV en mode ajout ('a') et écrire les en-têtes si nécessaire
with open(csv_filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Nom du fichier", "Aire (px2)", "Aire (mm2)", "Diametre (px)", "Diametre (mm)", "Centre X", "Centre Y", "Circularit", "Ratio axes"])

# Charger l'image
img_path = "1-Images_gouttes/1215.jpg"
img = cv2.imread(img_path)
img_name = os.path.basename(img_path)

# Convertir en HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Définir les seuils pour détecter la goutte
bound_lower = np.array([0, 0, 0])
bound_upper = np.array([180, 255, 50])
mask_black = cv2.inRange(hsv_img, bound_lower, bound_upper)

# Améliorer le masque avec des opérations morphologiques
kernel = np.ones((7, 7), np.uint8)
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)

# Trouver les contours
contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copier l'image pour affichage
output = img.copy()

if contours:
    # Prendre le plus grand contour (supposé être la goutte)
    c = max(contours, key=cv2.contourArea)

    # 🔹 Surface (Aire en px² et conversion mm²)
    area_px = cv2.contourArea(c)
    area_mm2 = area_px * pixel_to_mm2

    # 🔹 Centre de masse
    M = cv2.moments(c)
    cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0, 0)

    # 🔹 Circularité
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0

    # 🔹 Diamètre équivalent
    diameter_px = 2 * np.sqrt(area_px / np.pi)  # Basé sur la surface
    diameter_mm = diameter_px * pixel_to_mm

    # 🔹 Ratio des axes (ellipse)
    if len(c) >= 5:
        ellipse = cv2.fitEllipse(c)
        (xc, yc), (major_axis, minor_axis), angle = ellipse
        axis_ratio = major_axis / minor_axis if minor_axis > 0 else 1
        cv2.ellipse(output, ellipse, (0, 255, 0), 2)
    else:
        axis_ratio = 1

    # Ajouter les résultats au CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([img_name, area_px, area_mm2, diameter_px, diameter_mm, cx, cy, circularity, axis_ratio])

    # Afficher les résultats
    print(f"📂 Résultats enregistrés dans {csv_filename}")
    print(f"📏 Surface: {area_px:.2f} px² ({area_mm2:.2f} mm²)")
    print(f"📍 Centre: ({cx}, {cy})")
    print(f"⭕ Circularité: {circularity:.4f}")
    print(f"📏 Diamètre: {diameter_px:.2f} px ({diameter_mm:.2f} mm)")
    print(f"📏 Ratio axes: {axis_ratio:.2f}")

    # Dessiner les contours et le centre de masse
    cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
    cv2.circle(output, (cx, cy), 5, (255, 0, 0), -1)  # Centre en bleu

# Affichage final
cv2.imshow("Goutte détectée", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

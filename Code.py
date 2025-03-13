import cv2
import numpy as np

# Charger l'image
img = cv2.imread("1-Images_gouttes/0387.jpg")

# Convertir en HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Définir les seuils pour détecter la goutte
bound_lower = np.array([0, 0, 0])      # Noir
bound_upper = np.array([180, 255, 50]) # Limite du noir

# Créer le masque pour extraire la goutte
mask_black = cv2.inRange(hsv_img, bound_lower, bound_upper)

# Améliorer le masque avec des opérations morphologiques
kernel = np.ones((7, 7), np.uint8)
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)

# Trouver les contours
contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copier l'image pour affichage
output = img.copy()

# Vérifier qu'un contour est détecté
if contours:
    # Prendre le plus grand contour (supposé être la goutte)
    c = max(contours, key=cv2.contourArea)

    # ---- 🔹 1. Surface (Aire) ----
    area = cv2.contourArea(c)

    # ---- 🔹 2. Diamètre équivalent (disque de même surface) ----
    diameter_eq = np.sqrt(4 * area / np.pi)

    # ---- 🔹 3. Centre de masse (Moment d'inertie) ----
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # ---- 🔹 4. Circularité ----
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # ---- 🔹 5. Ratio des axes de l’ellipse ----
    if len(c) >= 5:  # Vérifier si l'on peut ajuster une ellipse
        ellipse = cv2.fitEllipse(c)
        (xc, yc), (major_axis, minor_axis), angle = ellipse
        axis_ratio = major_axis / minor_axis if minor_axis > 0 else 1
        cv2.ellipse(output, ellipse, (0, 255, 0), 2)  # Dessiner l'ellipse
    else:
        axis_ratio = 1

    # Afficher les résultats
    print(f"📏 Surface: {area:.2f} px²")
    print(f"⚫ Diamètre équivalent: {diameter_eq:.2f} px")
    print(f"📍 Centre de masse: ({cx}, {cy})")
    print(f"⭕ Circularité: {circularity:.4f}")
    print(f"📏 Ratio grand axe/petit axe: {axis_ratio:.2f}")

    # Dessiner les contours et le centre de masse
    cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
    cv2.circle(output, (cx, cy), 5, (255, 0, 0), -1)  # Centre en bleu

# Affichage final
cv2.imshow("Goutte détectée", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
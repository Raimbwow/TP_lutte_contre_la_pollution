import cv2

img = cv2.imread("1-Images_gouttes/0387.jpg")  # Charger l'image
cv2.imshow("Image", img)  # Afficher
cv2.waitKey(0)  # Attendre une touche pour fermer
cv2.destroyAllWindows()  # Fermer la fenêtre
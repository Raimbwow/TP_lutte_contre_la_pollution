import cv2
import numpy as np

img = cv2.imread("1-Images_gouttes/0387.jpg")  # Charger l'image
cv2.imshow("Image", img)  # Afficher
cv2.waitKey(0)  # Attendre une touche pour fermer
cv2.destroyAllWindows()  # Fermer la fenêtre



hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
bound_lower = np.array([0, 0, 0])
bound_upper = np.array([180, 255, 50])

mask_black = cv2.inRange(hsv_img, bound_lower, bound_upper)
mask_inv = cv2.bitwise_not(mask_black)  # Inverse le masque
kernel = np.ones((7, 7), np.uint8)

mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)

seg_img = cv2.bitwise_and(img, img, mask=mask_black)
contours, hier = cv2.findContours(
    mask_black.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
output = img.copy()
output[mask_black > 0] = (0, 0, 255)  # Colorie la goutte en rouge (BGR)
seg_img = cv2.bitwise_and(img, img, mask=mask_black)

#output = cv2.drawContours(seg_img, contours, -1, (0, 0, 255), 3)

cv2.imshow("Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
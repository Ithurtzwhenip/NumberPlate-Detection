import cv2
import numpy as np
import imutils
import easyocr
import csv
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.show()

# Apply bilateral filter and edge detection
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) # Noise reduction
edged = cv2.Canny(bfilter, 30, 200) # Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
plt.show()

# Find contours
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Detect number plate
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

# Mask the number plate region
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()

# Crop the number plate region
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.show()

# Read the number plate
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

# Extract number plate text
if result:
    number_plate_text = result[0][-2]
else:
    number_plate_text = "Not detected"

# Display the number plate text
print("Detected Number Plate:", number_plate_text)

# Write the detected number plate to a CSV file
csv_filename = 'detected_number_plates.csv'
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([number_plate_text])

print("Detected number plate saved in:", csv_filename)

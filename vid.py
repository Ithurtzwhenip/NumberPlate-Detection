import cv2
import numpy as np
import imutils
import easyocr
import csv

# Path to the video file
video_file = 'vid.mp4'

# Initialize video capture object
cap = cv2.VideoCapture(video_file)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize CSV file for saving detected number plates
csv_filename = 'detected_number_plates.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame Number', 'Number Plate'])

frame_number = 0

# Loop over video frames
while True:
    ret, frame = cap.read()  # Read frame from video capture

    if not ret:
        break  # Break the loop if no frame is read

    frame_number += 1

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter and edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

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

    # Extract number plate text
    number_plate_text = "Not detected"
    if location is not None:
        # Mask the number plate region
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)

        # Crop the number plate region
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        # Read the number plate using EasyOCR
        result = reader.readtext(cropped_image)
        if result:
            number_plate_text = result[0][-2]

    # Save detected number plate to CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frame_number, number_plate_text])

    # Display frame with number plate text
    cv2.putText(frame, text=number_plate_text, org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.imshow('Frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

print("Detected number plates saved in:", csv_filename)

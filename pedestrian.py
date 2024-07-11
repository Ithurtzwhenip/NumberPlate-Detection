import cv2

# Create video capturing object
cap = cv2.VideoCapture('walking.mp4')

# Load body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Loop once video is successfully loaded
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Grayscale image for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow("Pedestrian Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

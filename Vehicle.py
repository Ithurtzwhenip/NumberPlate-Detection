import cv2

# Create video capturing object
cap = cv2.VideoCapture('cars.mp4')

# Load vehicle classifier
vehicle_detector = cv2.CascadeClassifier('haarcascade_car.xml')

# Get frame rate of the video
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Loop once video is successfully loaded
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Grayscale image for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to vehicle classifier
    vehicles = vehicle_detector.detectMultiScale(gray, 1.4, 2)

    # Extract bounding boxes for any vehicles identified
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow("Vehicle Detector", frame)
    # Delay to match the frame rate of the original video
    cv2.waitKey(int(1000 / frame_rate))

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

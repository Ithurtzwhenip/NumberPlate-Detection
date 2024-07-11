import numpy as np
import cv2
from deepface import DeepFace

gun_cascade = cv2.CascadeClassifier('cascade.xml')
cap = cv2.VideoCapture(0)

gun_exist = False

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    guns = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=25, minSize=(100, 100))

    face_obj = DeepFace.extract_faces(frame, enforce_detection=False)

    for face in face_obj:
        fx, fy, fw, fh = [list(face["facial_area"].values())[i] for i in range(4)]
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 225, 255), 5)


    for (x, y, w, h) in guns:
        gun_exist = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

    cv2.imshow('FEED', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if gun_exist:
    print('Guns are detected')
else:
    print('No guns are detected')

cap.release()
cv2.destroyAllWindows()
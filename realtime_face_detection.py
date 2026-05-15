import cv2
import os
import numpy as np

# ------------------- Load Haar Cascade for face detection -------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------- Load dataset images -------------------
dataset_path = "dataset"
faces_db = {}
for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    faces_db[person] = []
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces_db[person].append(cv2.resize(img, (200, 200)))  # consistent size

# ------------------- Start webcam -------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Press ESC or Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi_gray = gray[y:y+h, x:x+w]

        # --------- Simple face recognition ---------
        name = "Unknown"
        max_match = 0
        for person, imgs in faces_db.items():
            for img in imgs:
                try:
                    img_resized = cv2.resize(img, (w, h))
                    diff = np.sum((face_roi_gray - img_resized) ** 2)
                    match_score = 1 / (1 + diff)
                    if match_score > max_match:
                        max_match = match_score
                        name = person
                except:
                    continue

        # --------- Draw rectangle and label ---------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{name}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or Q
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
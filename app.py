
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# LOAD MODELS

eye_model = load_model("models/eye_model.h5")
mouth_model = load_model("models/mouth_model.h5")

eye_labels = ['closed', 'open']
mouth_labels = ['no_yawn', 'yawn']


# PREPROCESS

def preprocess(img):
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# FATIGUE LOGIC

def fatigue_level(eye, mouth):
    if eye == "closed":
        return "Severe Fatigue"
    elif mouth == "yawn":
        return "Mild Fatigue"
    else:
        return "Alert"


# FACE DETECTOR

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# WEBCAM

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        
        # EYE REGION
        
        eye_img = face[int(h*0.25):int(h*0.45), int(w*0.25):int(w*0.75)]
        eye_input = preprocess(eye_img)

        eye_pred = eye_model.predict(eye_input, verbose=0)
        eye_conf = np.max(eye_pred)
        eye_label = eye_labels[np.argmax(eye_pred)]

        if eye_label == "open" and eye_conf < 0.6:
            eye_label = "closed"

        
        # MOUTH (FULL FRAME)
        
        mouth_input = preprocess(frame)
        mouth_pred = mouth_model.predict(mouth_input, verbose=0)
        mouth_label = mouth_labels[np.argmax(mouth_pred)]

        
        # FATIGUE
        
        fatigue = fatigue_level(eye_label, mouth_label)

        
        # DRAW
        
        color = (0,255,0)

        if fatigue == "Mild Fatigue":
            color = (0,255,255)
        elif fatigue == "Severe Fatigue":
            color = (0,0,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        cv2.putText(frame, f"Eye: {eye_label}", (x, y-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Mouth: {mouth_label}", (x, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, fatigue, (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
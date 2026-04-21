import cv2
import numpy as np
from tensorflow.keras.models import load_model


# LOAD MODELS

eye_model = load_model("models/eye_model.h5")
mouth_model = load_model("models/mouth_model.h5")

eye_labels = ['closed', 'open']
mouth_labels = ['no_yawn', 'yawn']


# PREPROCESS FUNCTION

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


# TEST WITH IMAGE



img = cv2.imread("test2.jpg")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("No face detected ❌")
else:
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]

        # ✅ EYE → use cropped region
        eye_img = face[int(h*0.25):int(h*0.45), int(w*0.25):int(w*0.75)]
        eye_input = preprocess(eye_img)
        # eye_pred = eye_model.predict(eye_input)
        # eye_label = eye_labels[np.argmax(eye_pred)]
        eye_pred = eye_model.predict(eye_input)
        eye_conf = np.max(eye_pred)
        eye_label = eye_labels[np.argmax(eye_pred)]

        # 🔥 Confidence correction
        if eye_label == "open" and eye_conf < 0.6:
            eye_label = "closed"

        # ✅ MOUTH → use FULL IMAGE (not cropped)
        mouth_input = preprocess(img)
        mouth_pred = mouth_model.predict(mouth_input)
        mouth_label = mouth_labels[np.argmax(mouth_pred)]

        fatigue = fatigue_level(eye_label, mouth_label)

        print("Eye:", eye_label)
        print("Mouth:", mouth_label)
        print("Fatigue Level:", fatigue)
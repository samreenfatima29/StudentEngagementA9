import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chart_django_project.settings')
django.setup()



import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import csv
from chartApp.models import Student

def overlay_results(frame, label):
    cv2.rectangle(frame, (10,10), (300,50), (0, 0, 0), -1)
    cv2.putText(frame, label, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), cv2.LINE_AA)
    return frame

class_labels = ['Not Engaged: Looking Away', 'Not Engaged: Bored', 'Engaged: Confused', 'Not Engaged: Drowsy', 'Engaged: engaged', 'Engaged: Frustrated']
model = load_model('facial_recognizer.h5')

cap = cv2.VideoCapture(0)
last_prediction_time = time.time()

with open('output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Time', 'Prediction'])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    if current_time - last_prediction_time >= 1:

        # Frame preprocessing
        frame_resized = cv2.resize(frame, (300, 300))
        frame_normalized = frame_resized.astype(np.float32) / 255.0

        predict_proba = model.predict(np.expand_dims(frame_normalized, axis=0))
        predict_class = np.argmax(predict_proba)
        predict_label = class_labels[predict_class]

        # Create a new Student object and save it to the database
        student = Student(
            confused=0,
            lookingaway=0,
            drowsy=0,
            frustated=0,
            engaged=0,
            bored=0
)
        if predict_label == 'Not Engaged: Looking Away':
            student.lookingaway = 1
        elif predict_label == 'Not Engaged: Bored':
            student.bored = 1
        elif predict_label == 'Engaged: Confused':
            student.confused = 1
        elif predict_label == 'Not Engaged: Drowsy':
            student.drowsy = 1
        elif predict_label == 'Engaged: engaged':
            student.engaged = 1
        elif predict_label == 'Engaged: Frustrated':
            student.frustated = 1
        student.save()

        with open('output.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), predict_label])

        last_prediction_time = current_time

        frame_overlay = overlay_results(frame, predict_label)
        cv2.imshow('Facial Expression Recognition', frame_overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# import os
# import django

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chart_django_project.settings')
# django.setup()
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import time
# import csv

# def overlay_results(frame, label):
#     cv2.rectangle(frame, (10,10), (300,50), (0, 0, 0), -1)
#     cv2.putText(frame, label, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), cv2.LINE_AA)
#     return frame

# class_labels = ['Not Engaged: Looking Away', 'Not Engaged: Bored', 'Engaged: Confused', 'Not Engaged: Drowsy', 'Engaged: engaged', 'Engaged: Frustrated']
# model = load_model('facial_recognizer.h5')

# cap = cv2.VideoCapture(0 )
# last_prediction_time = time.time() 

# with open('output.csv', 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['Time', 'Prediction'])

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     current_time = time.time()  

#     if current_time - last_prediction_time >= 1:

#         # Frame preprocessing
#         frame_resized = cv2.resize(frame, (300, 300))
#         frame_normalized = frame_resized.astype(np.float32) / 255.0

#         predict_proba = model.predict(np.expand_dims(frame_normalized, axis = 0))
#         predict_class = np.argmax(predict_proba)
#         predict_label = class_labels[predict_class]

#         with open('output.csv', 'w', newline='') as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), predict_label])

#         last_prediction_time = current_time

#         frame_overlay = overlay_results(frame, predict_label)
#         cv2.imshow('Facial Expression Recognition', frame_overlay)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# import os
# import django

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chart_django_project.settings')
# django.setup()
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import time
# from chartApp.models import Student

# def overlay_results(frame, label):
#     cv2.rectangle(frame, (10,10), (300,50), (0, 0, 0), -1)
#     cv2.putText(frame, label, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), cv2.LINE_AA)
#     return frame

# class_labels = ['Not Engaged: Looking Away', 'Not Engaged: Bored', 'Engaged: Confused', 'Not Engaged: Drowsy', 'Engaged: engaged', 'Engaged: Frustrated']
# model = load_model('facial_recognizer.h5')

# cap = cv2.VideoCapture(0)
# last_prediction_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     current_time = time.time()

#     if current_time - last_prediction_time >= 1:

#         # Frame preprocessing
#         frame_resized = cv2.resize(frame, (300, 300))
#         frame_normalized = frame_resized.astype(np.float32) / 255.0

#         predict_proba = model.predict(np.expand_dims(frame_normalized, axis=0))
#         predict_class = np.argmax(predict_proba)
#         predict_label = class_labels[predict_class]

#         # Create a new Student object and save it to the database
#         student = Student(confused=0, lookingaway=0, drowsy=0, frustated=0, engaged=0, bored=0)

#         if predict_label == 'Not Engaged: Looking Away':
#             student.lookingaway = 1
#         elif predict_label == 'Not Engaged: Bored':
#             student.bored = 1
#         elif predict_label == 'Engaged: Confused':
#             student.confused = 1
#         elif predict_label == 'Not Engaged: Drowsy':
#             student.drowsy = 1
#         elif predict_label == 'Engaged: engaged':
#             student.engaged = 1
#         elif predict_label == 'Engaged: Frustrated':
#             student.frustated = 1

#         student.save()

#         last_prediction_time = current_time

#         frame_overlay = overlay_results(frame, predict_label)
#         cv2.imshow('Facial Expression Recognition', frame_overlay)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chart_django_project.settings')
django.setup()

# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# import time
# from chartApp.models import Student

# def overlay_results(frame, label):
#     cv2.rectangle(frame, (10,10), (300,50), (0, 0, 0), -1)
#     cv2.putText(frame, label, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), cv2.LINE_AA)
#     return frame

# class_labels = ['Not Engaged: Looking Away', 'Not Engaged: Bored', 'Engaged: Confused', 'Not Engaged: Drowsy', 'Engaged: engaged', 'Engaged: Frustrated']
# model = load_model('facial_recognizer.h5')

# cap = cv2.VideoCapture(0)
# last_prediction_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     current_time = time.time()

#     if current_time - last_prediction_time >= 1:

#         # Frame preprocessing
#         frame_resized = cv2.resize(frame, (300, 300))
#         frame_normalized = frame_resized.astype(np.float32) / 255.0

#         predict_proba = model.predict(np.expand_dims(frame_normalized, axis=0))
#         predict_class = np.argmax(predict_proba)
#         predict_label = class_labels[predict_class]

#         # Create a new Student object and save it to the database
#         student = Student(confused=0, lookingaway=0, drowsy=0, frustated=0, engaged=0, bored=0)

#         if predict_label == 'Not Engaged: Looking Away':
#             student.lookingaway = 1
#         elif predict_label == 'Not Engaged: Bored':
#             student.bored = 1
#         elif predict_label == 'Engaged: Confused':
#             student.confused = 1
#         elif predict_label == 'Not Engaged: Drowsy':
#             student.drowsy = 1
#         elif predict_label == 'Engaged: engaged':
#             student.engaged = 1
#         elif predict_label == 'Engaged: Frustrated':
#             student.frustated = 1

#         student.save()

#         last_prediction_time = current_time

#         frame_overlay = overlay_results(frame, predict_label)
#         cv2.imshow('Facial Expression Recognition', frame_overlay)
        
#         print(f"Prediction: {predict_label}")
#         print(f"Student object saved: {student}")

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import Counter
from chartApp.models import Student

def overlay_results(frame, label):
    cv2.rectangle(frame, (10,10), (300,50), (0, 0, 0), -1)
    cv2.putText(frame, label, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), cv2.LINE_AA)
    return frame

class_labels = ['Not Engaged: Looking Away', 'Not Engaged: Bored', 'Engaged: Confused', 'Not Engaged: Drowsy', 'Engaged: engaged', 'Engaged: Frustrated']
model = load_model('facial_recognizer.h5')

cap = cv2.VideoCapture(0)
threshold = 0.5  # Threshold for considering an attribute
attribute_counts = Counter()  # Counter to maintain counts of attributes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame preprocessing
    frame_resized = cv2.resize(frame, (300, 300))
    frame_normalized = frame_resized.astype(np.float32) / 255.0

    predict_proba = model.predict(np.expand_dims(frame_normalized, axis=0))
    predict_class = np.argmax(predict_proba)
    predict_label = class_labels[predict_class]

    # Update attribute counts
    attribute_counts[predict_label] += 1

    frame_overlay = overlay_results(frame, predict_label)
    cv2.imshow('Facial Expression Recognition', frame_overlay)
    
    print(f"Prediction: {predict_label}")

    # Check if the window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Check if any attribute count exceeds the threshold
        max_label, max_count = attribute_counts.most_common(1)[0]
        total_count = sum(attribute_counts.values())
        if max_count / total_count >= threshold:
            # Create a new Student object and save it to the database
            student = Student()

            if max_label == 'Not Engaged: Looking Away':
                student.lookingaway = 1
            elif max_label == 'Not Engaged: Bored':
                student.bored = 1
            elif max_label == 'Engaged: Confused':
                student.confused = 1
            elif max_label == 'Not Engaged: Drowsy':
                student.drowsy = 1
            elif max_label == 'Engaged: engaged':
                student.engaged = 1
            elif max_label == 'Engaged: Frustrated':
                student.frustated = 1

            student.save()
            print(f"Student object saved: {student}")

        break  # Exit the loop if the window is closed

cap.release()
cv2.destroyAllWindows()


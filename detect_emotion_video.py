import cv2
import numpy as np
import imutils
import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import os
import math
import sys
from threading import Timer
import shutil
import time
import keras
import numpy as np
import pandas as pd
from keras.models import model_from_json
import csv



def detect_and_predict_face(frame, faceNet, face_emotion_model, threshold):
    global detections
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            locs.append((startX, startY, endX, endY))
            preds.append(face_emotion_model.predict(face)[0].tolist())

    return (locs, preds)


MASK_MODEL_PATH = r'C:\Users\Acer\codes\live_fer\face\model\emotion_model.h5'
THRESHOLD = 0.5

print("[INFO] loading face detector model...")
prototxtPath = r'C:\Users\Acer\codes\live_fer\face\face_detector\deploy.prototxt'
weightsPath = r'C:\Users\Acer\codes\live_fer\face\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading emotion detector model...")
face_emotion_model = load_model(MASK_MODEL_PATH)

print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)

labels = ["happy", "neutral", "sad"]

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    original_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    (locs, preds) = detect_and_predict_face(frame, faceNet, face_emotion_model, THRESHOLD)
        
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        label = str(labels[np.argmax(pred)])
        if label == "happy":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 200, 50), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (0, 200, 50), 2)
            
        elif label == "neutral":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (255, 255, 255), 2)
            

        elif label == "sad":
            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 50, 200), 2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (0, 50, 200), 2)
            


    frame = cv2.resize(original_frame, (860, 490))
    cv2.imshow("Facial Expression", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    time.sleep(0.25)

cv2.destroyAllWindows()
vs.stop()

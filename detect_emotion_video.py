import cv2
import numpy as np
import imutils
import time
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import time
from utils import detect_and_predict_face





THRESHOLD = 0.5
model = 0
if model == 0:
    emotion_model = r"face\model\emotion_model.h5"
    labels = ["happy", "neutral", "sad"]
    IMG_SIZE = (224, 224)
    COLOR_TRANSFORMER = cv2.COLOR_BGR2RGB

else:
    emotion_model = "fer_model_from_scratch.h5"
    labels = ['surprise', 'fear', 'angry', 'neutral', 'sad', 'disgust', 'happy']
    IMG_SIZE = (48, 48)
    COLOR_TRANSFORMER = cv2.COLOR_BGR2GRAY



print("[INFO] loading face detector model...")
prototxtPath = r'C:\Users\Acer\codes\live_fer\face\face_detector\deploy.prototxt'
weightsPath = r'C:\Users\Acer\codes\live_fer\face\face_detector\res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading emotion detector model...")
face_emotion_model = load_model(emotion_model)

print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    original_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    (locs, preds) = detect_and_predict_face(frame, faceNet, face_emotion_model, THRESHOLD, IMG_SIZE, COLOR_TRANSFORMER)
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

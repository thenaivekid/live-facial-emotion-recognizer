import cv2 
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def detect_and_predict_face(frame, faceNet, face_emotion_model, threshold, img_size, color_transformer):
    """Detect faces in an input image frame, and predict emotions for each detected face.

    Parameters:
    - frame (numpy.ndarray): The input image frame in the form of a NumPy array.
    - faceNet: The pre-trained face detection model (OpenCV DNN module).
    - face_emotion_model: The pre-trained model for predicting emotions (TensorFlow model).
    - threshold (float): Confidence threshold for face detection. Faces with confidence
      below this threshold will be ignored.

    Returns:
    Tuple[List[Tuple[int, int, int, int]], List[List[float]]]:
        A tuple containing two lists:
        - The first list (locs) contains tuples representing the coordinates of each detected face
          in the format (startX, startY, endX, endY).
        - The second list (preds) contains lists representing the predicted emotions for each detected face.
    """
    global detections
    (h, w) = frame.shape[:2]
    # cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height), mean=(mean_R, mean_G, mean_B), swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
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
            face = cv2.cvtColor(face, color_transformer)
            
            face = cv2.resize(face, img_size)
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            locs.append((startX, startY, endX, endY))
            preds.append(face_emotion_model.predict(face)[0].tolist())

    return (locs, preds)
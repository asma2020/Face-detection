import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#modpreel structure: https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt
#-trained weights: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
s = 'D:/Edge DL/deploy.prototxt'

detector = cv2.dnn.readNetFromCaffe(s, "D:/Edge DL/res10_300x300_ssd_iter_140000.caffemodel")

detector.getLayerNames()

# Open the video file
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    # print(frame.shape)
    final  = np.zeros((480, 840, 3))
    detected_faces = []
    if not ret:
        break

    base_img = frame.copy()

    original_size = frame.shape
    target_size = (300, 300)
    print("original image size: ", original_size)

    frame = cv2.resize(frame, target_size)

    aspect_ratio_x = (original_size[1] / target_size[1])
    aspect_ratio_y = (original_size[0] / target_size[0])
    print("aspect ratios x: ",aspect_ratio_x,", y: ", aspect_ratio_y)

    frame.shape

    #detector expects (1, 3, 300, 300) shaped input
    imageBlob = cv2.dnn.blobFromImage(image = frame)
    #imageBlob = np.expand_dims(np.rollaxis(image, 2, 0), axis = 0)

    detector.setInput(imageBlob)
    detections = detector.forward()

    detections[0][0].shape

    detections_df = pd.DataFrame(detections[0][0]
        , columns = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])

    detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
    detections_df = detections_df[detections_df['confidence'] >= 0.20]

    for i, instance in detections_df.iterrows():
        confidence_score = str(round(100*instance["confidence"], 2))+" %"

        left = int(instance["left"] * 300)
        bottom = int(instance["bottom"] * 300)
        right = int(instance["right"] * 300)
        top = int(instance["top"] * 300)

        cv2.putText(base_img, confidence_score, (int(left*aspect_ratio_x), int(top*aspect_ratio_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(base_img, (int(left*aspect_ratio_x), int(top*aspect_ratio_y)), (int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)), (255, 255, 255), 1) #draw rectangle to main image


        # akbari
        (h, w) = frame.shape[:2]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype(int)
        face = frame[startY:endY, startX:endX]
        if face.shape[0] > 0 and face.shape[1] > 0:
            detected_faces.append(face)
    # Display the frame with detected faces
    cv2.imshow('Face Detection', base_img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

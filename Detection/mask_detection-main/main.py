import cv2
import os
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imageio import imsave

cmd = argparse.ArgumentParser()
cmd.add_argument('-i','--image', help='image path required')
args = vars(cmd.parse_args())
path_weight = './face_detector/res10_300x300_ssd_iter_140000.caffemodel'
path_prototext = './face_detector/deploy.prototxt'
path_model = './face_detector/mask_detector.model'
confidence_percentage = 0.5
net = cv2.dnn.readNet(path_prototext, path_weight)
model = load_model(path_model)
image = cv2.imread(args['image'])
(image_height, image_width) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1, (300, 300), (104, 177, 123))
net.setInput(blob)
detection = net.forward()
for i in range(0, detection.shape[2]):
    confidence = detection[0, 0, i, 2]
    if confidence > confidence_percentage:
        box = detection[0, 0,i,3:7] * np.array([image_width, image_height, image_width, image_height])
        (startX, startY, endX, endY) = box.astype(int)
        print(startX, startY, endX, endY)
        # (startX, startY) = (max(0,startX), max(0,startY))
        # (endX, endY) = (min(image_width - 1, endX) , min(image_height - 1, endY))
        face = image[startY: endY, startX: endX]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        (mask , no_mask) = model.predict(face)[0]
        print(mask , no_mask)
        if mask > no_mask:
            predicted_label = 'avec mask'
            color = (0, 255, 0)
        else:
            predicted_label = 'sans mask'
            color = (255, 0, 0)
        cv2.putText(image, predicted_label, (startX, startY) , cv2.FONT_HERSHEY_COMPLEX, 1, color)
        cv2.rectangle(image, (startX, startY), (endX, endY), color)

cv2.imshow('detect_face', image)
cv2.waitKey(0)
imsave("resultat/resultat1.png", image)
cropped=image[startY:endY , startX:endX]
cv2.imshow('cropped', cropped)
imsave("resultat/resultat2.png", cropped)
imsave("resultat/resultat3.png", cv2.resize(cv2.imread("resultat/resultat2.png",0), (125, 150)))

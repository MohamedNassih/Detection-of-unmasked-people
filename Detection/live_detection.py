import cv2
import os
import numpy as np
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
from imageio.v2 import imsave, imread
from scipy import linalg
import time
import glob
parser = argparse.ArgumentParser()
parser.add_argument('-i','--image', help='image path required')
args = vars(parser.parse_args())
path_weight = 'mask_detection-main/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
path_prototext = 'mask_detection-main/face_detector/deploy.prototxt'
path_model = 'mask_detection-main/face_detector/mask_detector.model'
confidence_percentage = 0.5
net = cv2.dnn.readNet(path_prototext, path_weight)
model = load_model(path_model)
cap = cv2.VideoCapture(0)
k = True
while True:
    _,image=cap.read()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detection = net.forward()
    for i in range(0, detection.shape[2]):
        box = detection[0, 0,i,3:7] * np.array([w, h, w, h])
        (sX, sY, eX, eY) = box.astype(int)
        confidence = detection[0, 0, i, 2]
        if confidence > confidence_percentage:
            (sX, sY) = (max(0, sX), max(0, sY))
            (eX, eY) = (min(w-1, eX) , min(h-1, eY))
            face = image[sY: eY, sX: eX]
            face = cv2.resize(face, (224, 224))
            cropped = face
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask , no_mask) = model.predict(face)[0]
            print(mask , no_mask)
            if mask > no_mask:
                predicted_label = 'avec mask'
                color = (0, 255, 0)
                k = True
            else:
                predicted_label = 'sans mask'
                color = (255, 0, 0)
                k = False
            cv2.putText(image, predicted_label, (sX, sY) , cv2.FONT_HERSHEY_COMPLEX, 1, color)
            cv2.rectangle(image, (sX, sY), (eX, eY), color)
    cv2.imshow('detect face', image)
    imsave("mask_detection-main/resultat/resultat1.png", image)
    if k == False :
        imsave("mask_detection-main/resultat/resultat2.png", cropped)
        cv2.imshow('A trouver', cropped)
        imsave("myFaceReco/a_tester/1.png", cv2.resize(cv2.imread("mask_detection-main/resultat/resultat2.png",0), (125, 150)))
        pngs = glob.glob('myFaceReco/gallery/*.png')
        image = imread(pngs[0])
        heigh = len(image[:,0])
        length= len(image[0])
        imgs = np.array([imread(i).flatten() for i in pngs])
        moyenne = np.mean(imgs,0)
        phi=imgs-moyenne
        covreduit=np.dot(phi, phi.transpose())
        valp,vectpreduit=np.linalg.eig(covreduit)
        vectp=np.dot(phi.transpose(),vectpreduit)
        poids=np.dot(phi,vectp)  
        #for i in range(len(imgs)):
            #imsave("myFaceReco/eigenface/eigenface"+str(i)+".png",vectp[:,i].reshape(heigh,length))
        #for k in range(len(imgs)):
            #for i in range(len(imgs)):
                #recon=moyenne+np.dot(poids[k,:i],vectp[:,:i].transpose())
                #imsave("myFaceReco/reconst/img_"+str(k)+"_"+str(i)+".png",recon.reshape(heigh,length))
        t1=time.time()
        image_a_trouver = np.array(imread("myFaceReco/a_tester/1.png").flatten())
        phi2=image_a_trouver-moyenne
        poids2= np.dot(phi2,vectp)
        dist = np.min((poids-poids2)**2,axis=1)
        indiceImg = np.argmin(dist)
        print("Image de : "+pngs[indiceImg])
        cv2.imshow("c'est trouvé", imgs[indiceImg].reshape(heigh,length))
        #if mindist <=2.0:
            #print( "concordance !")
        #else:
            #print( "pas de concordance !")
        t2=time.time()
        ex=t2-t1
        print(f'le temps d\'éxecution est : {ex:.2}ms')
    cv2.waitKey(0)
cap.release()

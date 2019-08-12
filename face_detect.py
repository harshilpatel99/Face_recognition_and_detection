# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:10:58 2019

@author: I'm Harshil
"""

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle
data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
Training_Data, Labels, IDs = [], [], []
names=['Harshil','Heli']
with open('available_persons','rb') as f:
        known_persons = pickle.load(f)

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ID = int(onlyfiles[i].split('_',1)[0])
   
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
    IDs.append(ID)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(IDs, dtype=np.int32))



print("Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('G:/OpenCV/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

def face_detector(image, size = 0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        cv2.putText(image, "Face Not Found!", (10,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
      
        return image,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(image, (x,y),(x+w,y+h),(0,0,0),2)
        
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200)) 
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            value, result = model.predict(face)
            v=str(value)
            temp_name = known_persons.loc[known_persons['Id'] == v, 'Name'].item()
         
            size = cv2.getTextSize(temp_name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image, (x,y+h),(x+w,y+h-size[0][1]),(0,0,0),-1)
            
            
            if result < 500:                
                confidence = int(100*(1-(result)/300))
                display_string = str(confidence)+'% True'
            cv2.putText(image,display_string,(10,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            if confidence > 80:            
                if (value):   
                    cv2.putText(image,temp_name, (x+2,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow('Face Cropper', image)    
            else:
                cv2.putText(image, "Stranger!", (x+2,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255),1, cv2.LINE_AA)
                cv2.imshow('Face Cropper', image)
        except:        
            cv2.putText(image, "Face Not Found! with exception", (10,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass


cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    ret=cv2.flip(ret,1)
    face_detector(frame)
    

    if cv2.waitKey(1)==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
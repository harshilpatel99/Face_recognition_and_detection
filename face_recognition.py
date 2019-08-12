# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:26:02 2019

@author: I'm Harshil
"""

import cv2
import numpy as np
import pandas as pd
import pickle

face_classifier = cv2.CascadeClassifier('G:/OpenCV/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')


def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[x:x+w,y:y+h]

    return cropped_face

ID = input("Enter id(Integer numbers only): ")
Name = input("Enter name: ")
flag = 1
with open('available_persons','rb') as f:
    df = pickle.load(f)
if(ID and Name):    
    ids = df['Id']
    for i in ids:
        if(i==ID):
            print("Id already exists. Try some different integer Id!")
            flag = 0
            break
    if(flag):
        cap = cv2.VideoCapture(0)
        count = 0
        
        while True:
            ret, frame = cap.read()
            frame=cv2.flip(frame,1)
            ret=cv2.flip(ret,1)
            if face_extractor(frame) is not None:
                count+=1
                face = cv2.resize(face_extractor(frame),(200,200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
                file_name_path = 'faces/'+str(ID)+'_'+str(count)+'_'+Name+'.jpg'
                cv2.imwrite(file_name_path,face)
        
                cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.imshow('Face Cropper',face)
            else:
                print("Face not Found")
                pass
        
            if cv2.waitKey(1)==13 or count==100:
                break
       
        with open('available_persons','rb') as f:
            df = pickle.load(f)
        df = df.append({"Id": ID, "Name": Name}, ignore_index=True)
        with open('available_persons','wb') as f:
            pickle.dump(df,f)        
        cap.release()
        cv2.destroyAllWindows()
        print('Colleting Samples Complete!!!')
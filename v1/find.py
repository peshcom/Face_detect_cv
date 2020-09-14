#! /usr/bin/env python3

import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, img = cap.read()
    #img = cv2.flip(img, -1) # Переворот изображения

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Переводим в черно - белое
    
    faces = faceCascade.detectMultiScale(
        gray,     			# Изображение
        scaleFactor=1.2,	# это параметр, определяющий размер изображения при каждой шкале изображения. Он используется для создания масштабной пирамиды
        minNeighbors=5,     # параметр, указывающий, сколько соседей должно иметь каждый прямоугольник кандидата, чтобы сохранить его. 
        # Более высокое число дает более низкие ложные срабатывания.
        minSize=(20, 20)	# минимальный размер прямоугольника, который считается лицом.
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import time

cap=cv2.VideoCapture('https://192.168.1.34:8080/video')
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('resouse.avi',fourcc,50.0,(1920,1080))
t0 = time.time() #start
t1 = time.time() #ประกาศตัวแปร

while t1-t0<75: #5*(15s)
    t1 = time.time()
    _,frame=cap.read()
    frame=cv2.resize(frame,(1920,1080))
    cv2.imshow('window', frame)
    out.write(frame) #payh :  C:\Users\L\Downloads\project\time\resouse.avi

    #ทางลัด
    key=cv2.waitKey(1)
    if key==ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
#pylint:disable=no-member

import numpy as np
import cv2 as cv
from datetime import date
import urllib.request as urllib2
from time import sleep
import RPi.GPIO as GPIO
#from gpiozero import LED

#ledR=LED(20)
#ledG=LED(21)
ledR=20
ledG=21

prevdate2=2
prevdate3=3
prevdate4=4


GPIO.setmode(GPIO.BCM)
GPIO.setup(ledG,GPIO.OUT)
GPIO.setup(ledR,GPIO.OUT)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
today = date.today()
prevdate="06/06/2023"

myapi ="E3SZQJ5C5EI6LP29"
baseURL = 'https://api.thingspeak.com/update?api_key=%s' % myapi
#baseURL = 'https://api.thingspeak.com/update?api_key=%s' % myapi
At = 20



# people = ['person1', 'person2', 'person3','person4']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

people = ['anas', 'anubhav', 'karan','Mintu']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')


capture=cv.VideoCapture(0)

while(True):
      
    
 ret, img = capture.read()

 #img = capture.read()
 GPIO.output(ledG,GPIO.LOW)
 GPIO.output(ledR,GPIO.HIGH)
 gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#  cv.imshow('Person', gray)

 #  Detect the face in the image
 faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 6)

 for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    if (confidence < 45):
      cv.putText(img, "Unknown", (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
      #GPIO.output(ledG,GPIO.HIGH)
      #GPIO.output(ledG,GPIO.LOW)
      #ledR.on()
      #sleep(1)
    else:
     cv.putText(img, str(people[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
     
     #ledG.on()
     
    if(prevdate!= today):
     

     if(str(people[label])=="anas"):
      f = urllib2.urlopen(baseURL +"&field1=%s" %At)
      prevdate = today
      f.close()
      GPIO.output(ledG,GPIO.HIGH)
      GPIO.output(ledR,GPIO.LOW)
      sleep(2)
      #pic=cv.imread('C:/Users/DELL/Desktop/open cv/saves2/anas/1.jpg')
      #cv.imshow('attended', pic)
      # sleep(2)
      #cv.waitKey(1000)
      #capture.release()
    if(prevdate2!= today):
     if(str(people[label])=="anubhav"):
      f = urllib2.urlopen(baseURL +"&field2=%s" %At) 
      prevdate2 = today
      f.close()
      GPIO.output(ledG,GPIO.HIGH)
      GPIO.output(ledR,GPIO.LOW)
      sleep(2)
      #pic=cv.imread('C:/Users/DELL/Desktop/open cv/saves2/anas/1.jpg')
      #cv.imshow('attended', pic)
      # sleep(2)
      #cv.waitKey(1000)
      #capture.release()
      # capture.release()
    if(prevdate3!= today):
     if(str(people[label])=="karan"):
      f = urllib2.urlopen(baseURL +"&field3=%s" %At) 
      prevdate3 = today
      f.close()
      GPIO.output(ledG,GPIO.HIGH)
      GPIO.output(ledR,GPIO.LOW)
      sleep(2)
      
     # print (f.read())
    if(prevdate4!= today):
     if(str(people[label])=="Mintu"):
      f = urllib2.urlopen(baseURL +"&field4=%s" %At) 
      prevdate4 = today
      f.close()
      GPIO.output(ledG,GPIO.HIGH)
      GPIO.output(ledR,GPIO.LOW)
      sleep(2)
     
     
     


 cv.imshow('Detected Face', img)
 
 #cv.waitKey(1000)
 if cv.waitKey(1)==ord('q'):
  break

capture.release()
cv.destroyAllWindows()
#pylint:disable=no-member

import os
import cv2 as cv
import numpy as np

#people = ['person1', 'person2', 'person3','person4']

people = ['anas', 'anubhav','karan','Mintu']

#DIR = r'/home/pi/Desktop/face recognition/faces/train'

DIR = r'/home/pi/Desktop/face recognition/Saves2/saves2'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        # print('loop1')
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            # print('loop2')
            # print(img)
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                #print('loop3')
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                #print(features)
                #print(('x,y,w,h'),(x, y ,w ,h))
                #print(faces_roi)
                labels.append(label)
                #print(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

#face_recognizer = cv.createLBPHFaceRecognizer()

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

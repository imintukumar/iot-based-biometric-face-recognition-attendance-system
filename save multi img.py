import cv2 as cv
import time 
#import numpy as np
#capture = cv.VideoCapture('videos/dog.mp4')

capture = cv.VideoCapture(0)
count = 0
while True:
    
 isTrue  , Frame = capture.read()
 cv.imshow('video',Frame)
  
 gray = cv.cvtColor(Frame, cv.COLOR_BGR2GRAY)

 for i in range(1,21):

  # cv.SaveImage('pic{:>05}.jpg'.format(i), img)
  #time.sleep(1)
  if (count ==0): 
    cv.imwrite('/home/pi/Desktop/face recognition/Saves2/saves2/Mintu/img{:>16}.jpg'.format(i),Frame)
    print(i)
    
    if (i==20):
     count=1
     break
    


 if cv.waitKey(20) & 0xff==ord('d'):
    break 
capture.release()
cv.destroyAllWindows()


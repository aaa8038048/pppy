# -*- coding: utf-8 -*-

import cv2.cv as cv  
import cv2  
from cv2 import VideoCapture  
      
#cv.NamedWindow("W1", cv.CV_WINDOW_AUTOSIZE)  
cv.NamedWindow("W1",cv.CV_WINDOW_NORMAL)

cv.ResizeWindow("W1", 600, 600)
      
      
capture = cv.CaptureFromCAM(0) 

      
    
      
def repeat():  
    frame=cv.LoadImage('face9.jpg')
        
    #frame = cv.QueryFrame(capture)

    image_size = cv.GetSize(frame)
	
          
    greyscale = cv.CreateImage(image_size, 8, 1)

    cv.CvtColor(frame, greyscale, cv.CV_BGR2GRAY)

    storage = cv.CreateMemStorage(0)
          
    cv.EqualizeHist(greyscale, greyscale)#���Ҷ�ͼ��ֱ��ͼ���⻯��ò�ƿ���ʹ�Ҷ�ͼ����Ϣ�����٣��ӿ����ٶ�  
    
    #��ͼ��ָ���
         
   
        # detect objects  
    cascade = cv.Load('C:\opencv2.3.1\opencv\data\haarcascades\haarcascade_frontalface_alt2.xml')
    #����Intel��˾��ѵ����  
      
        #���ͼƬ�е�������������һ��������������Ϣ�Ķ���faces  
    faces = cv.HaarDetectObjects(greyscale, cascade, storage, 1.2, 2,
                                 cv.CV_HAAR_DO_CANNY_PRUNING,
                                 (100, 100))  
      
        #�����������λ�õ�����  
    for (x,y,w,h) , n in faces:
       # print x,y
      
        cv.Rectangle(frame, (x,y), (x+w,y+h), (0,128,0),2)#����Ӧλ�ñ�ʶһ������ �߿�����(0,0,255)��ɫ 20���
          
        cv.ShowImage("W1", greyscale)#��ʾ���б߿��ͼƬ
          
    cv.ShowImage("W1", frame)  

	  
      
     
while True:  
    repeat()  
    c = cv.WaitKey(10)  
    if c == 27:  
        cv2.VideoCapture(0).release()  
        cv2.destroyWindow("W1")  
        break
#if __name__=='__main__':
#	repeat()
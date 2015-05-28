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
          
    cv.EqualizeHist(greyscale, greyscale)#将灰度图像直方图均衡化，貌似可以使灰度图像信息量减少，加快检测速度  
    
    #画图像分割线
         
   
        # detect objects  
    cascade = cv.Load('C:\opencv2.3.1\opencv\data\haarcascades\haarcascade_frontalface_alt2.xml')
    #加载Intel公司的训练库  
      
        #检测图片中的人脸，并返回一个包含了人脸信息的对象faces  
    faces = cv.HaarDetectObjects(greyscale, cascade, storage, 1.2, 2,
                                 cv.CV_HAAR_DO_CANNY_PRUNING,
                                 (100, 100))  
      
        #获得人脸所在位置的数据  
    for (x,y,w,h) , n in faces:
       # print x,y
      
        cv.Rectangle(frame, (x,y), (x+w,y+h), (0,128,0),2)#在相应位置标识一个矩形 边框属性(0,0,255)红色 20宽度
          
        cv.ShowImage("W1", greyscale)#显示互有边框的图片
          
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
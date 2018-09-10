import cv2
import numpy as np
import math

class drawMov:
	def __init__(self,target):
		self.tx,self.ty,self.bx,self.by = int(target[0]),int(target[1]),int(target[2]),int(target[3])
		self.top=self.ty
		self.bottom = self.by
		self.left = self.tx
		self.right = self.bx
		self.width = self.right - self.left
		self.height = self.bottom - self.top
		self.center = self.getCenter(target) 

	def getCenter(self,bbox):
		center = [(int(bbox[2])-int(bbox[0]))/2,(int(bbox[3])-int(bbox[1]))/2]
		return center

	def drawLine(self,img,bbox):
		moveCenter=self.getCenter(bbox)
		cv2.line(img,self.center,moveCenter,(255,0,0),2)

	def getAngle(self,bbox):
		moveCenter=self.getCenter(bbox)
		distance=math.sqrt((moveCenter[0]-self.center[0])**2+(moveCenter[1]-self.center[1])**2)
		cTheta=(moveCenter[1]-self.center[1])/(distance+10e-5)
		angle=math.degrees(math.acos(cTheta))
		return angle


cv2.resizeWindow('test',600,480)
while(True):
	mask=np.zeros((600,800,3),np.uint8)
	#(120,280 250,350)
	cv2.rectangle(mask,(0,0),(250,350),(0,255,0),2)
	cv2.circle(mask,(int(mask.shape[1]/2),int(mask.shape[0]/2)+100),2,(255,0,0),-1)
	cv2.imshow('video',mask)
	print(mask.shape[0]/2,int(mask.shape[1]/2))
	cv2.waitKey(0)

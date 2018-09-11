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
		center = [int((bbox[2]+bbox[0])/2),int((bbox[3]+bbox[1])/2)]
		return center

	def drawLine(self,img,bbox):
		moveCenter=self.getCenter(bbox)
		cv2.line(img,(self.center[0],self.center[1]),(moveCenter[0],moveCenter[1]),(255,0,0),2)

	def getAngle(self,bbox):
		moveCenter=self.getCenter(bbox)
		distance=math.sqrt((moveCenter[0]-self.center[0])**2+(moveCenter[1]-self.center[1])**2)
		cTheta=(moveCenter[1]-self.center[1])/(distance+10e-5)
		angle=math.degrees(math.acos(cTheta))
		return angle

	def drawCenter(self,img):
		cv2.circle(img,(int(self.center[0]),int(self.center[1])),2,(255,0,0),-1)




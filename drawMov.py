#coding=utf8
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
		return [int((bbox[2]+bbox[0])/2),int((bbox[3]+bbox[1])/2)]

	def drawLine(self,img):
		#moveCenter=self.getCenter(bbox)
		moveCenter=self.getStandardCenter(img)
		cv2.line(img,(self.center[0],self.center[1]),(moveCenter[0],moveCenter[1]),(255,0,0),2)

	def getAngle(self,img):
		moveCenter=self.getStandardCenter(img)
		distance=math.sqrt((moveCenter[0]-self.center[0])**2+(moveCenter[1]-self.center[1])**2)
		cTheta=(moveCenter[1]-self.center[1])/(distance+10e-5)
		angle=math.degrees(math.acos(cTheta))
		return angle

	def drawCenter(self,img):
		cv2.circle(img,tuple(self.center),2,(255,0,0),-1)

	def adjustCenter(self,img,stack):
		# right + , front +, vertical
		roll, vertical = 0,0
		stackLR=stack
		standardCenter=self.getStandardCenter(img)
		cv2.circle(img,tuple(standardCenter),2,(0,0,255),-1)
		#cv2.putText(img,"center",tuple(standardCenter),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
		roll=self.center[0]-standardCenter[0]
		vertical=self.center[1]-standardCenter[1]
		print([roll,vertical])
		if roll < -1 :
			roll = -1
			stackLR -= 1
		elif roll > 1 :
			roll = 1
			stackLR += 1
		if vertical < -1:
			vertical = 1
		elif vertical > 1 :
			vertical=-1
		return roll,vertical,stackLR

	def getStandardCenter(self,img):
		return [int(img.shape[1]/2),int(img.shape[0]/2+100)]

	def getStandardBox(self,img):
		standardCenter=self.getStandardCenter(img)
		inBox=(80,200)
		outBox=(130,300)
		cv2.rectangle(img,(int(standardCenter[0]-inBox[0]/2),int(standardCenter[1]-inBox[1]/2)),
			(int(standardCenter[0]+inBox[0]/2),int(standardCenter[1]+inBox[1]/2)),(0,0,255),1)
		cv2.rectangle(img,(int(standardCenter[0]-outBox[0]/2),int(standardCenter[1]-outBox[1]/2)),
			(int(standardCenter[0]+outBox[0]/2),int(standardCenter[1]+outBox[1]/2)),(0,0,255),1)
		return inBox,outBox

	def adjustBox(self,img):
		pitch = 0
		inBox,outBox=self.getStandardBox(img)
		if self.width<inBox[0] and self.height<inBox[1]:
			pitch = 1
		elif self.width>outBox[0] and self.height>outBox[1]:
			pitch = -1
		return pitch


	def adjPos(self,img,target,angleStack):
		roll,pitch,yaw,vertical=0,0,0,0
		angle=0
		stack=angleStack
		pos=[roll,pitch,yaw,vertical]
		cv2.putText(img, "Following The Target", (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
		message=[]
		self.__init__(target)
		pitch = self.adjustBox(img)
		roll,vertical,stack=self.adjustCenter(img,stack)
		
		if roll < 0:
			message.append('left')
		elif roll > 0 :
			message.append('right')

		if vertical < 0 :
			message.append('down')
		elif vertical > 0 :
			message.append('up')

		if pitch > 0 :
			message.append('front')
		elif pitch < 0:
			message.append('back')

		if stack > 10 :
			angle = self.getAngle(img)
			stack = 0
			print('angle: ', angle)
		elif stack < -10:
			angle = -(self.getAngle(img))
			stack = 0
			print('angle: ', angle)

		print(message)

		return stack





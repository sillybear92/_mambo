#coding=utf8
import cv2
import numpy as np
import math
from pyparrot.Minidrone import Mambo
from pyparrot.scripts import findMinidrone

class drawMov:
	def __init__(self):
		self.tx,self.ty,self.bx,self.by = None,None,None,None
		self.top=None
		self.bottom = None
		self.left = None
		self.right = None
		self.width = None
		self.height = None
		self.center = None
		self.mamboAddr,self.mamboName = None,None
		self.mambo = None
		self.droneCheck=False
		self.droneBattery=None
		self.inBox=(260,380) #(130,300)->(250,300)->(260,380)
		self.outBox=(330,450) #(200,380)->(330,380)->(330,450)

	def update(self):
		self.mambo.smart_sleep(0.01)
		self.droneBattery=int(self.mambo.sensors.battery)
		print("Battery:",self.droneBattery,"%   State:",self.mambo.sensors.flying_state)


	def setTarget(self,target):
		self.tx,self.ty,self.bx,self.by = int(target[0]),int(target[1]),int(target[2]),int(target[3])
		self.top=self.ty
		self.bottom = self.by
		self.left = self.tx
		self.right = self.bx
		self.width = self.right - self.left
		self.height = self.bottom - self.top
		self.center = self.getCenter(target)
		
	def droneConnect(self):
		self.mamboAddr,self.mamboName = findMinidrone.getMamboAddr()
		self.mambo = Mambo(self.mamboAddr, use_wifi=False)
		self.droneCheck=self.mambo.connect(num_retries=3)
		print("Drone Connect: ",self.droneCheck)
		self.mambo.smart_sleep(2)
		self.mambo.ask_for_state_update()
		self.mambo.set_max_tilt(1)

	def droneStart(self):
		print('take off')
		self.mambo.safe_takeoff(5)


	def droneStop(self):
		if not self.mambo.sensors.flying_state == 'landed':
			self.mambo.safe_land(5)
		try:
			self.mambo.disconnect()
		except:
			print("No Ground Cam!!")
		print("Complete to Stop the Drone!")
	
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

	def adjustVertical(self):
		vertical=0
		ih=self.inBox[1]
		oh=self.outBox[1]
		vertical=self.top
		if vertical < oh:
			vertical = 20
		elif vertical > ih :
			vertical=-20
		return vertical

	def adjustCenter(self,img,stack,yawTime):
		# right + , front +, vertical
		roll, yaw = 0,0,0
		angle=0
		yawCount=yawTime
		stackLR=stack
		standardCenter=self.getStandardCenter(img)
		#cv2.putText(img,"center",tuple(standardCenter),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
		roll=self.center[0]-standardCenter[0]
		if roll < -1 :
			roll = -20
			stackLR -= 1
		elif roll > 1 :
			roll = 20
			stackLR += 1
		if yawCount <-1:
			yaw=-50
			yawCount += 1
		elif yawCount >1 :
			yaw = 50
			yawCount -= 1
		if stackLR > 20 :
			angle = self.getAngle(img)
			stackLR = 0
			print('angle: ', angle)
			yawCount=int(angle/7)
		elif stackLR < -20:
			angle = -(self.getAngle(img))
			stackLR = 0
			print('angle: ', angle)
			yawCount=int(angle/7)
		return roll,yaw,stackLR,yawCount

	def getStandardCenter(self,img):
		return [int(img.shape[1]/2),int(img.shape[0]/2+80)]
		#80->150->0->80

	def drawStandardBox(self,img):
		standardCenter=self.getStandardCenter(img)
		cv2.circle(img,tuple(standardCenter),2,(0,0,255),-1)
		cv2.rectangle(img,(int(standardCenter[0]-self.inBox[0]/2),int(standardCenter[1]-self.inBox[1]/2)),
			(int(standardCenter[0]+self.inBox[0]/2),int(standardCenter[1]+self.inBox[1]/2)),(0,0,255),1)
		cv2.rectangle(img,(int(standardCenter[0]-self.outBox[0]/2),int(standardCenter[1]-self.outBox[1]/2)),
			(int(standardCenter[0]+self.outBox[0]/2),int(standardCenter[1]+self.outBox[1]/2)),(0,0,255),1)

	def adjustBox(self,img):
		pitch = 0
		if self.width*self.height<self.inBox[0]*self.inBox[1]:
			pitch = 30
		elif self.width*self.height>self.outBox[0]*self.outBox[1]:
			pitch = -30
		return pitch


	def adjPos(self,img,target,angleStack,yawTime):
		roll,pitch,yaw,vertical,duration=0,0,0,0,0.1
		angle=0
		self.drawStandardBox(img)
		stack=angleStack
		cv2.putText(img, "Following The Target", (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
		self.setTarget(target)
		vertical=self.adjustVertical()
		if vertical == 0:
			pitch = self.adjustBox(img)
			roll,yaw,stack,yawTime=self.adjustCenter(img,stack,yawTime)
		pos=[roll,pitch,yaw,vertical]
		if pos==[0,0,0,0]:
			stack=0
		else:
			self.mambo.fly_direct(roll=roll, pitch=pitch, yaw=yaw, vertical_movement=vertical, duration=duration)
			print('Roll:',roll,' Pitch:',pitch,' Yaw:',yaw,' Vertical:',vertical)

		return stack,yawTime





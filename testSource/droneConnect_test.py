import os
import drawMov
import cv2
import numpy as np


def main():
	mov = drawMov.drawMov()
	while not mov.droneCheck:
		mov.droneConnect()
	mask = np.zeros((50,50,3),dtype=np.uint8)
	while(True):
		cv2.imshow("PressKey",mask)
		mov.update()
		key=cv2.waitKey(10)
		if ord('q')==key:
			mov.droneStop()
			exit(0)
		elif ord('p')==key:
			mov.droneStart()
		elif ord('w')==key:
			mov.mambo.fly_direct(roll=0,pitch=100,yaw=0,vertical_movement=0,duration=0.1)
		elif ord('s')==key:
			mov.mambo.fly_direct(roll=0,pitch=-100,yaw=0,vertical_movement=0,duration=0.1)
		elif ord('a')==key:
			mov.mambo.fly_direct(roll=-50,pitch=0,yaw=0,vertical_movement=0,duration=0.1)
		elif ord('d')==key:
			mov.mambo.fly_direct(roll=50,pitch=0,yaw=0,vertical_movement=0,duration=0.1)
		elif ord('i')==key:
			mov.mambo.fly_direct(roll=0,pitch=0,yaw=0,vertical_movement=50,duration=0.1)
		elif ord('k')==key:
			mov.mambo.fly_direct(roll=0,pitch=0,yaw=0,vertical_movement=-50,duration=0.1)
		elif ord('j')==key:
			mov.mambo.fly_direct(roll=0,pitch=0,yaw=-50,vertical_movement=-50,duration=0.1)
		elif ord('l')==key:
			mov.mambo.fly_direct(roll=0,pitch=0,yaw=50,vertical_movement=-50,duration=0.1)
if __name__=='__main__':
	main()

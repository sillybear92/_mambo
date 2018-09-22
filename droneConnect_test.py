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
if __name__=='__main__':
	main()

'''
	http://blog.naver.com/samsjang/220504633218
'''
import cv2
import numpy as np
import sys
from mss import mss
import time

def dp_fps(img,prevTime):
	# Display FPS
	curTime=time.time()
	sec=curTime - prevTime
	prevTime=curTime
	fps = 1/(sec)
	cv2.putText(img,"FPS : %0.1f"%fps,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	return prevTime

def main():
	#Capture x,y position
	mon={'top':300, 'left':600, 'width':100, 'height':200}
	#window ScreenCapture
	sct=mss()
	while True:
		img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
		hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
		lowGreen=np.array([30,100,100])
		upGreen=np.array([55,255,255])
		lowRed=np.array([115,100,100])
		upRed=np.array([125,255,255])
		maskGreen=cv2.inRange(hsv,lowGreen,upGreen)
		maskRed=cv2.inRange(hsv,lowRed,upRed)
		gr=cv2.bitwise_and(img,img,mask=maskGreen)
		rd=cv2.bitwise_and(img,img,mask=maskRed)
		cv2.imshow('original',img)
		cv2.imshow('green',gr)
		cv2.imshow('red',rd)
		pp=rd.sum()-gr.sum()
		if pp>0:
			print(u'빨간색')
		elif pp<0:
			print(u'초록색')
		else:
			print(u'감지 할 수 없음')
		if ord('q')==cv2.waitKey(10):
			cv2.destroyAllWindows()
			exit()

if __name__=='__main__':
	main()


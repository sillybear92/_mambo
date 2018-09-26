import stt
from multiprocessing import Process
from detect_client import netInfo
import socket
import cv2
import numpy as np
import sys
import time
import imutils
import pickle


def draw_rectangle(img,result):
	n_detect=[]
	labels=['hand','person']
	for obj in result:
		label = obj['label']
		if not label in [l for l in labels]:
			continue
		confidence = obj['confidence']
		top_x = obj['topleft']['x']
		top_y = obj['topleft']['y']
		bottom_x = obj['bottomright']['x']
		bottom_y = obj['bottomright']['y']
		label = obj['label']
		if label=='hand':
			cv2.rectangle(img,(top_x, top_y),(bottom_x, bottom_y), (0, 255, 0),2)
			cv2.putText(img, label+' - ' + str(  "{0:.0f}%".format(confidence * 100) ),
				(bottom_x, top_y-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)
			n_detect.append([top_x,top_y,bottom_x,bottom_y])
		elif label == 'person':
			n_detect.append([top_x,top_y,bottom_x,bottom_y])
	return n_detect

# Display FPS
def dp_fps(img,prevTime):
	curTime=time.time()
	sec=curTime - prevTime
	prevTime=curTime
	fps = 1/(sec)
	cv2.putText(img,"FPS : %0.1f"%fps,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	return prevTime

def main():
	client=netInfo()
	client.setServer('192.168.0.14',6666)
	p=Process(target=stt.run)
	p.start()
	prevTime=0
	while True:
		img,result = client.sendData(b'hdg')
		if result==-1:
			continue
		gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		mask=np.ones_like(img,np.uint8)
		# Display FPS
		prevTime=dp_fps(img,prevTime)
		hand=draw_rectangle(img,result)
		cv2.imshow('video',img)
		if ord('q')==cv2.waitKey(10):
			exit()
		if (p.is_alive()):
			continue
		print(u'STT와 연결 끊김')

if __name__=='__main__':
	main()


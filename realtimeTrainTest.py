from darkflow.net.build import TFNet
import cv2
import numpy as np
import os
import sys
import time


def dp_fps(img,prevTime):
	# Display FPS
	curTime=time.time()
	sec=curTime - prevTime
	prevTime=curTime
	fps = 1/(sec)
	cv2.putText(img,"FPS : %0.1f"%fps,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	return prevTime

def draw_rec(img,result):
	n_hand=[]
	for obj in result:
		confidence = obj['confidence']
		top_x = obj['topleft']['x']
		top_y = obj['topleft']['y']
		bottom_x = obj['bottomright']['x']
		bottom_y = obj['bottomright']['y']
		label = obj['label']
		#test
		#cv2.rectangle(img,(top_x, top_y),(bottom_x, bottom_y), (0, 255, 0),2)
		#cv2.putText(img, label+' - ' + str(  "{0:.0f}%".format(confidence * 100) ),(bottom_x, top_y-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)
		# Person Recognition & Boxing
		if(confidence>0.1):
			cv2.rectangle(img,(top_x, top_y),(bottom_x, bottom_y), (0, 255, 0),2)
			cv2.putText(img, label+' - ' + str(  "{0:.0f}%".format(confidence * 100) ),(bottom_x, top_y-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)
		if label == 'person':
			n_hand.append([top_x,top_y,bottom_x,bottom_y])
	return n_hand

hp="F:\\AnacondaProjects"
optionPerson={"pbLoad":hp+"\\darkflow-master\\built_graph\\yolo180905.pb",
 "metaLoad":hp+ "\\darkflow-master\\built_graph\\yolo180905.meta",
 "threshold":0.4, "gpu":0.7}
personTf=TFNet(optionPerson)
img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)


prevTime=0

while(True):
	img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
	result=personTf.return_predict(img)
	person=draw_rec(img,result)
	# Display FPS
	prevTime=dp_fps(img,prevTime)
	cv2.imshow('Image', img)
	cv2.waitKey(1)




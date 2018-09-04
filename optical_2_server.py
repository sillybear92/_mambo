#!/home/lee/anaconda2/bin/python
#coding=utf8
'''
	shape_detect(): OpenCV shape detection 
	-> https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
'''
from darkflow.net.build import TFNet
import cv2
import numpy as np
import urllib
import time
from PIL import Image
from mss import mss
import imutils

#Flow Option
hp="C:\\Users\\lee\\AnacondaProjects"
'''options={"pbLoad":hp+"\\darkflow-master\\built_graph\\new-hand-voc.pb",
 "metaLoad":hp+ "\\darkflow-master\\built_graph\\new-hand-voc.meta",
 "threshold":0.4, "gpu":0.7}'''
options={"pbLoad":hp+"\\darkflow-master\\built_graph\\hand-yolo.pb",
 "metaLoad":hp+ "\\darkflow-master\\built_graph\\hand-yolo.meta",
 "threshold":0.4, "gpu":0.7}

# Draw_rec and person motion recognition save
def draw_rec(img,result):
	n_hand=[]
	for obj in result:
		confidence = obj['confidence']
		top_x = obj['topleft']['x']
		top_y = obj['topleft']['y']
		bottom_x = obj['bottomright']['x']
		bottom_y = obj['bottomright']['y']
		label = obj['label']
		# Person Recognition & Boxing
		if(confidence>0.1 and label=='hand'):
			cv2.rectangle(img,(top_x, top_y),(bottom_x, bottom_y), (0, 255, 0),2)
			cv2.putText(img, label+' - ' + str(  "{0:.0f}%".format(confidence * 100) ),(bottom_x, top_y-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)
			n_hand.append([top_x,top_y,bottom_x,bottom_y])
	return n_hand

def dp_fps(img,prevTime):
	# Display FPS
	curTime=time.time()
	sec=curTime - prevTime
	prevTime=curTime
	fps = 1/(sec)
	cv2.putText(img,"FPS : %0.1f"%fps,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	return prevTime

#Center of Box - point Draw
def center_Box(hand):
	new_track=[]
	for t_x,t_y,b_x,b_y in hand:
		c_x=(t_x+b_x)/2
		c_y=(t_y+b_y)/2
		new_track.append([[c_x,c_y]])
	return new_track

#Distance of Hand - point Distance
def distance_Box(old_track,new_track):
	n_track=[]
	len_old = len(old_track)
	len_new = len(new_track)
	if len_old==0:
		n_track=new_track
	elif len_new==0:
		n_track=old_track
	else:
		old_point=np.float32([old_tr[-1] for old_tr in old_track]).reshape(-1,1,2)
		new_point=np.float32([new_tr[-1] for new_tr in new_track]).reshape(1,-1,2)
		mask=abs(old_point-new_point).max(-1)<60
		p_old,p_new=np.where(mask==True)
		for p_x,p_y in zip(p_old,p_new):
			old_track[p_x].append(new_track[p_y][-1])
			if len(old_track[p_x])>60:
				del old_track[p_x][0]
		z_index=np.concatenate(np.where(mask.sum(axis=0)==0))
		for zz in z_index:
			old_track.append(new_track[zz])
			if len(old_track)>6:
				del old_track[0]
		n_track=old_track
	return n_track

def shape_detect(mask,rec_info,target_hand):
	grayMask=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
	blurred=cv2.GaussianBlur(grayMask,(5,5),0)
	thresh=cv2.threshold(blurred,60,255,cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts=cnts[0] if imutils.is_cv2() else cnts[1]
	for c in cnts:
		shape="not_detect"
		m=cv2.moments(c)
		try:
			cX=int(m["m10"]/m["m00"])
			cY=int(m["m01"]/m["m00"])
		except:
			continue
		peri = cv2.arcLength(c,True)
		approx=cv2.approxPolyDP(c,0.04*peri,True)
		if len(approx) == 4:
			shape = "rectangle"
			rec_point=abs(np.array([rec[-1] for rec in rec_info]).reshape(-1,1,2)-np.array([cX,cY]).reshape(1,-1,2)).max(-1)<10
			p_1,p_2=np.where(rec_point==True)
			if len(p_1)==0:
				rec_info.append([[cX,cY]])
			else:
				for x,y in zip(p_1,p_2):
					rec_info[x].append([cX,cY])
					if len(rec_info[x])>10:
						if len(target_hand)==0:
							target_hand.append(c[-1][0])
						del rec_info[x][0]
		else:
			shape = "not_detect"
		cv2.drawContours(mask,[c],-1,(0,255,0),2)
		cv2.putText(mask,shape,(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
	return rec_info,target_hand


		

#Capture x,y position
mon={'top':150, 'left':150, 'width':800, 'height':600}
#window ScreenCapture
sct=mss()

if __name__ == '__main__':
	tfnet=TFNet(options)
	print("============ Video Start ============")
	prevTime=0
	track=[]
	hand=[]
	rec_info=[]
	target_hand=[]
	cnt=0
	while(True):
		img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
		gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		mask=np.ones_like(img,np.uint8)
		result=tfnet.return_predict(img)
		# Display FPS
		prevTime=dp_fps(img,prevTime)
		hand=draw_rec(img,result)
		track=distance_Box(track,center_Box(hand))
		cv2.polylines(mask,([np.int32(tr) for tr in track]),False,(255,255,0),3)
		if not len(target_hand)==0:
			cv2.putText(img,"Target",(target_hand[-1][0],target_hand[-1][1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
		#detect_shape
		rec_info,target_hand=shape_detect(mask,rec_info,target_hand)
		nImg=cv2.add(img,mask)
		#cv2.fillPoly(img,([np.int32(tr) for tr in track]),(0,255,0))
		cv2.imshow('video',nImg)
		if ord('q')==cv2.waitKey(10):
			exit(0)
	print('== Turn over ==')


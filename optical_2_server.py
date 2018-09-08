#!/home/lee/anaconda2/bin/python
#coding=utf8
'''
	shape_detect(): OpenCV shape detection 
	-> https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
	create tracker : Multiple Object Tracking using OpenCV 
	-> https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
'''
from darkflow.net.build import TFNet
import cv2
import numpy as np
import urllib
import time
from PIL import Image
from mss import mss
import imutils

#TF Options
hp="F:\\AnacondaProjects"
'''options={"pbLoad":hp+"\\darkflow-master\\built_graph\\new-hand-voc.pb",
 "metaLoad":hp+ "\\darkflow-master\\built_graph\\new-hand-voc.meta",
 "threshold":0.4, "gpu":0.7}'''
optionHand={"pbLoad":hp+"\\darkflow-master\\cfg\\tartget-yolo.cfg",
 "load":hp+ "\\darkflow-master\\bin\\yolo.weights",
 "gpu":0.7, "train","trainer":"adam",""}
optionPerson={"pbLoad":hp+"\\darkflow-master\\built_graph\\yolo.pb",
 "metaLoad":hp+ "\\darkflow-master\\built_graph\\yolo.meta",
 "threshold":0.4, "gpu":0.7}

# Draw_rec and person motion recognition save
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
		cv2.rectangle(img,(top_x, top_y),(bottom_x, bottom_y), (0, 255, 0),2)
		cv2.putText(img, label+' - ' + str(  "{0:.0f}%".format(confidence * 100) ),
			(bottom_x, top_y-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)
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

#Center of Box - point Draw
def center_Box(obj):
	new_track=[]
	for t_x,t_y,b_x,b_y in obj:
		c_x=(t_x+b_x)/2
		c_y=(t_y+b_y)/2
		new_track.append([[c_x,c_y]])
	return new_track

#Distance of Hand - point Distance
def distance_Box(old_track,new_track):
	n_track=[]
	len_old = len(old_track)
	len_new = len(new_track)
	# before Hand detection position list -> None
	if len_old==0:
		n_track=new_track
	# now Hand detection position list -> None
	elif len_new==0:
		n_track=old_track
	else:
		old_point=np.float32([old_tr[-1] for old_tr in old_track]).reshape(-1,1,2)
		new_point=np.float32([new_tr[-1] for new_tr in new_track]).reshape(1,-1,2)
		''' befor Hand detection position - now Hand detection position < 60 :: bigger distance < 60
			-> Same Hand '''
		mask=abs(old_point-new_point).max(-1)<60
		p_old,p_new=np.where(mask==True)
		# Same Hand to list
		for p_x,p_y in zip(p_old,p_new):
			old_track[p_x].append(new_track[p_y][-1])
			if len(old_track[p_x])>70:
				del old_track[p_x][0]
		''' ex) old_point.shape(3,1,2) * new_point.shape(2,1,2)
			=> mask.shape(3,2)
			mask.sum(axis=0)==0 => new Hand detection'''
		z_index=np.concatenate(np.where(mask.sum(axis=0)==0))
		for zz in z_index:
			old_track.append(new_track[zz])
			if len(old_track)>6:
				del old_track[0]
		n_track=old_track
	return n_track

# Shape from Hand moving line
def shape_detect(img,mask,rec_info,targetOn,tracker):
	grayMask=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
	blurred=cv2.GaussianBlur(grayMask,(5,5),0)
	thresh=cv2.threshold(blurred,60,255,cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts=cnts[0] if imutils.is_cv2() else cnts[1]
	target_hand=[]
	target=[]
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
			rec_point=abs(np.array([rec[-1] for rec in rec_info]).reshape(-1,1,2)\
				-np.array([cX,cY]).reshape(1,-1,2)).max(-1)<10
			p_1,p_2=np.where(rec_point==True)
			if len(p_1)==0:
				rec_info.append([[cX,cY]])
			else:
				for x,y in zip(p_1,p_2):
					rec_info[x].append([cX,cY])
					if len(rec_info[x])>50:
						if not targetOn:
							target_hand.append(c[-1][0])
							targetOn=get_target(img,tracker,target_hand)
						del rec_info[x][0]
		else:
			shape = "not_detect"
		cv2.drawContours(mask,[c],-1,(0,255,0),2)
		cv2.putText(mask,shape,(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
	return rec_info,targetOn

def preconnect_Net(mon):
	img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
	result=handTf.return_predict(img)
	result=personTf.return_predict(img)

def get_target(img,tracker,target_hand):
	target=[]
	targetFlag=0
	result=personTf.return_predict(img)
	person=draw_rectangle(img,result)
	neartarget=abs(np.array([[[tx,ty,bx,by]] for [tx,ty,bx,by] in person]).reshape(-1,4)\
	 - np.array([target_hand[-1][0],target_hand[-1][1],target_hand[-1][0],target_hand[-1][1]]).reshape(-1,4)).min(-1)
	near=neartarget[-1]
	for [tx,ty,bx,by] in person:
		if targetFlag:
			continue
		if (tx-near<target_hand[-1][0] and ty-near<target_hand[-1][1] 
			and bx+near>target_hand[-1][0] and by+near>target_hand[-1][1]):
			target.append([tx,ty,bx,by])
			targetFlag=1
	setTracker(img,tracker,target)
	return targetFlag

def setTracker(img,tracker,target):
	tx,ty,bx,by=target[-1][0],target[-1][1],target[-1][2],target[-1][3]
	bbox=(tx,ty,bx-tx,by-ty)
	ok=tracker.init(img,bbox)
	cv2.rectangle(img,(tx,ty),(bx,by),(0,255,255),3)
	cv2.putText(img,"Target",(bx, ty-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)

'''
Reference:https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/'''
def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
  return tracker

def updateTracker(img):
	ok,bbox=tracker.update(img)
	if ok:
		tx,ty,bx,by=int(bbox[0]),int(bbox[1]),int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])
		cv2.rectangle(img,(tx,ty),(bx,by),(0,255,255),3)
		cv2.putText(img,"Target",(bx, ty-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)
	else:
		cv2.putText(img, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)


if __name__ == '__main__':
	handTf=TFNet(optionHand)
	personTf=TFNet(optionPerson)
	#Capture x,y position
	mon={'top':150, 'left':150, 'width':800, 'height':600}
	#window ScreenCapture
	sct=mss()
	print("============ Video Start ============")
	prevTime=0
	track=[]
	hand=[]
	rec_info=[]
	targetOn=0
	trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
	tracker=createTrackerByName("KCF")
	preconnect_Net(mon)
	while(True):
		img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
		gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		mask=np.ones_like(img,np.uint8)
		if not targetOn:
			result=handTf.return_predict(img)
			# Display FPS
			prevTime=dp_fps(img,prevTime)
			hand=draw_rectangle(img,result)
			track=distance_Box(track,center_Box(hand))
			cv2.polylines(mask,([np.int32(tr) for tr in track]),False,(255,255,0),3)
			#detect_shape
			rec_info,targetOn=shape_detect(img,mask,rec_info,targetOn,tracker)
			img=cv2.add(img,mask)
		else:
			# Display FPS
			prevTime=dp_fps(img,prevTime)
			updateTracker(img)
		cv2.imshow('video',img)
		if ord('q')==cv2.waitKey(10):
			exit(0)
	print('== Turn over ==')


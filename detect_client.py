#coding=utf8
'''
	shape_detect(): OpenCV shape detection 
	-> https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
	create tracker : Multiple Object Tracking using OpenCV 
	-> https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/
'''

import socket
import cv2
import numpy as np
import sys
import time
import imutils
import pickle


class netInfo:
	def __init__(self):
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.host = sys.argv[1]
		self.port = int(sys.argv[2])
		self.server_address = (self.host,self.port)
		self.client_host = '0.0.0.0'
		self.client_port = 5001
		self.sock.bind((self.client_host,self.client_port))
		self.sock.settimeout(0.5)

	def sendData(self,message):
		self.sock.sendto(message,self.server_address)
		try:
			data,server = self.sock.recvfrom(65507)
		except socket.timeout as err:
			print("Timeout !! again !! ")
			return -1,-1
		print("Fragment size: {}".format(len(data)))
		if len(data) == 4:
			if(data == "FAIL"):
				return -1,-1
		unpick = pickle.loads(data)
		i = unpick["image"]
		array = np.frombuffer(i, dtype=np.dtype('uint8'))
		img = cv2.imdecode(array,1)
		try:
			result = unpick["result"]	
		except:
			result = None
		return img, result



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
def shape_detect(client,mask,rec_info,targetOn,tracker):
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
					if len(rec_info[x])>30:
						if not targetOn:
							target_hand.append(c[-1][0])
							targetOn=get_target(client,tracker,target_hand)
						del rec_info[x][0]
		else:
			shape = "not_detect"
		cv2.drawContours(mask,[c],-1,(0,255,0),2)
		cv2.putText(mask,shape,(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
	return rec_info,targetOn


def get_target(client,tracker,target_hand):
	target=[]
	targetFlag=0
	img,result = client.sendData("psg")
	if result==-1:
				return 
	person=draw_rectangle(img,result)
	neartarget=abs(np.array([[[tx,ty,bx,by]] for [tx,ty,bx,by] in person]).reshape(-1,4)\
	 - np.array([target_hand[-1][0],target_hand[-1][1],target_hand[-1][0],target_hand[-1][1]]).reshape(-1,4)).min(-1)
	near=neartarget[-1]+1
	for [tx,ty,bx,by] in person:
		if targetFlag:
			continue
		if (tx-near<target_hand[-1][0] and ty-near<target_hand[-1][1] 
			and bx+near>target_hand[-1][0] and by+near>target_hand[-1][1]):
			target.append([tx,ty,bx,by])
			targetFlag=1
	targetFlag=setTracker(img,tracker,target)
	return targetFlag

def setTracker(img,tracker,target):
	if target==None:
		return 0
	tx,ty,bx,by=target[-1][0],target[-1][1],target[-1][2],target[-1][3]
	bbox=(tx,ty,bx-tx,by-ty)
	ok=tracker.init(img,bbox)
	cv2.rectangle(img,(tx,ty),(bx,by),(0,255,255),3)
	cv2.putText(img,"Target",(bx, ty-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)
	return 1

'''
Reference:https://www.learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/'''
def createTrackerByName(trackerType):
  trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
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

def updateTracker(tracker,img):
	ok,bbox=tracker.update(img)
	if ok:
		tx,ty,bx,by=int(bbox[0]),int(bbox[1]),int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])
		cv2.rectangle(img,(tx,ty),(bx,by),(0,255,255),3)
		cv2.putText(img,"Target",(bx, ty-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)
	else:
		cv2.putText(img, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)



def main():
	client=netInfo()
	print ("server_address is ", client.server_address[0],client.server_address[1])
	prevTime=0
	track=[]
	hand=[]
	rec_info=[]
	targetOn=0
	tracker=createTrackerByName("KCF")
	while(True):
		if not targetOn:
			img,result = client.sendData("hdg")
			if result==-1:
				continue
			gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
			mask=np.ones_like(img,np.uint8)
			# Display FPS
			prevTime=dp_fps(img,prevTime)
			hand=draw_rectangle(img,result)
			track=distance_Box(track,center_Box(hand))
			cv2.polylines(mask,([np.int32(tr) for tr in track]),False,(255,255,0),3)
			#detect_shape
			rec_info,targetOn=shape_detect(client,mask,rec_info,targetOn,tracker)
			img=cv2.add(img,mask)
		else:
			img,result = client.sendData("get")
			if result==-1:
				continue
			# Display FPS
			prevTime=dp_fps(img,prevTime)
			updateTracker(tracker,img)
		cv2.imshow('video',img)
		if ord('q')==cv2.waitKey(10):
			exit(0)
	print('== Turn over ==')

if __name__=='__main__':
	main()

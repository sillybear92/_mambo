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
from multiprocessing import Process
from lib.netInfo2 import netInfo
from lib.resultTF import resultTF
from time import sleep



# Draw_rec and person motion recognition save
# YOLO 결과를 바탕으로 손의 경계박스 표현
def hand_rectangle(img,result):
	# 인식한 물체의 좌표를 담을 리스트
	n_detect=[]
	for obj in result:
		# 탐지한 Class 이름
		label = obj['label']
		# 신뢰도
		confidence = obj['confidence']
		# 왼쪽 위 (x,y) -> (top_x,top_y)
		top_x = obj['topleft']['x']
		top_y = obj['topleft']['y']
		# 오른쪽 아래 (x,y) -> (bottom_x,bottom_y)
		bottom_x = obj['bottomright']['x']
		bottom_y = obj['bottomright']['y']
		if label=='hand':
			# 위 좌표를 바탕으로 경계박스를 그림
			cv2.rectangle(img,(top_x, top_y),(bottom_x, bottom_y), (0, 255, 0),2)
			cv2.putText(img, label+' - ' + str(  "{0:.0f}%".format(confidence * 100) ),
				(bottom_x, top_y-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)
			n_detect.append([top_x,top_y,bottom_x,bottom_y])
	return n_detect

# YOLO 결과를 바탕으로 사람의 경계박스 표현
def person_rectangle(img,result):
	n_detect=[]
	for obj in result:
		label = obj['label']
		confidence = obj['confidence']
		top_x = obj['topleft']['x']
		top_y = obj['topleft']['y']
		bottom_x = obj['bottomright']['x']
		bottom_y = obj['bottomright']['y']
		if label == 'person':
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
# 손과 손의 거리를 계산 -> 동일손끼리 리스트로 묶음
def distance_Box(old_track,new_track):
	# 손과 손 사이 거리 (손 중앙을 기준)
	HAND_CENTER_DISTANCE=30
	# 한 화면에 표현할 수 있는 손의 수
	HAND_COUNT=8
	# hand list 요소들의 최대 Length 
	HAND_LINE_LENGTH=60
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
		#old_track, new_track: 3차원 리스트, [[hand1 [hand center0],[hand center1]...], [hand2 [hand center0],[hand center1]...], ...]
		old_point=np.float32([old_tr[-1] for old_tr in old_track]).reshape(-1,1,2)
		new_point=np.float32([new_tr[-1] for new_tr in new_track]).reshape(1,-1,2)
		''' befor Hand detection position - now Hand detection position < 60 :: bigger distance < 60
			-> Same Hand '''
		# 각각의 hand center 거리가 HAND_CENTER_DISTANCE 미만이면 True, 아니면 False 
		mask=abs(old_point-new_point).max(-1)<HAND_CENTER_DISTANCE
		# True 인 점 찾기, mask 의 index 0는 old_point index, index 1은 new_point index가 된다
		p_old,p_new=np.where(mask==True)
		# Same Hand to list
		for p_x,p_y in zip(p_old,p_new):
			old_track[p_x].append(new_track[p_y][-1])
			if len(old_track[p_x])>HAND_LINE_LENGTH:
				del old_track[p_x][0]
		''' ex) old_point.shape(3,1,2) * new_point.shape(2,1,2)
			=> mask.shape(3,2)
			mask.sum(axis=0)==0 => new Hand detection'''
		# 모든 값이 False인(모든 값의 합이 0) new_point 는 새로운 손 좌표라 할 수 있다
		z_index=np.concatenate(np.where(mask.sum(axis=0)==0))
		for zz in z_index:
			old_track.append(new_track[zz])
			if len(old_track)>HAND_COUNT:
				del old_track[0]
		n_track=old_track
	return n_track

# Shape from Hand moving line
# OpenCV Shape_detect 기반으로 사각형 탐지 및 타겟 인식으로 전환
def shape_detect(img,mask,rec_info,targetOn,tracker,tf_person):
	# 사각형과 사각형 사이의 거리 (같은 사각형인지 판별하는 기준)
	RECTANGLE_POINT_DISTANCE=10
	# 해당 영역의 사각형 갯수(타겟 인식으로 전환하는 기준) 
	SAME_RECTANGLE_POINT=20
	grayMask=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
	# 가우시안 함수를 적용하여 이미지 노이즈 제거
	blurred=cv2.GaussianBlur(grayMask,(5,5),0)
	# 이미지를 부드럽게 처리
	thresh=cv2.threshold(blurred,60,255,cv2.THRESH_BINARY)[1]
	# 도형 찾기
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
		# len(approx): 꼭지점의 개수 
		if len(approx) == 4:
			(x, y, w, h) = cv2.boundingRect(approx)
			# w*h : 도형의 넓이
			if w*h>100:
				shape = "rectangle"
				# rec_info: 이전 프레임까지의 사각형들의 집합, [cX,cY]: 현재 사각형의 중앙 점
				# 사각형과 사각형 사이의 거리로 같은 사각형인지 확인
				rec_point=abs(np.array([rec[-1] for rec in rec_info]).reshape(-1,1,2)\
					-np.array([cX,cY]).reshape(1,-1,2)).max(-1)<RECTANGLE_POINT_DISTANCE
				p_1,p_2=np.where(rec_point==True)
				# 새 사각형 탐지
				if len(p_1)==0:
					rec_info.append([[cX,cY]])
				else:
					for x,y in zip(p_1,p_2):
						rec_info[x].append([cX,cY])
						# 해당 영역의 사각형이 유지된 수가 기준보다 높으면 타겟 인식으로 전환 
						if len(rec_info[x])>SAME_RECTANGLE_POINT:
							if not targetOn:
								target_hand.append([c[-1][0][0],c[-1][0][1],c[-1][0][0],c[-1][0][1]])
								# 현재 이미지에서 YOLO 기반 사람 탐지 결과 
								result=tf_person.getBuffer(img)
								# get_target(): 추적할 Target의 좌표를 얻어내는 함수, Tracker 와 사람 좌표를 기반으로 동작한다
								targetOn,target=get_target(img,result,tracker,target_hand[0],targetOn)

							del rec_info[x][0]
		else:
			shape = "not_detect"
		cv2.drawContours(mask,[c],-1,(0,255,0),2)
		cv2.putText(mask,shape,(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
	return rec_info,targetOn,target

# 추적할 Target의 좌표를 얻어내는 함수, Tracker 와 사람 좌표를 기반으로 동작한다
def get_target(img,result,tracker,obj,targetOn=1):
	target=[]
	inbox=[]
	neartarget=np.array([]).reshape(-1,4)
	# 사람의 경계박스를 표현하고 사람 좌표를 얻는다
	person=person_rectangle(img,result)
	# YOLO에서 사람을 탐지하지 못했다면 Tracker 기반으로 target을 설정한다
	if len(person)==0:
		target=[obj]
	else:
		for [tx,ty,bx,by] in person:
			#array=np.array([tx,ty,bx,by])
			#Tracker에서 감지한 Target의 경계박스와 YOLO의 사람 좌표의 거리
			neartarget=np.append(neartarget,abs(np.array([tx,ty,bx,by])-np.array([obj[0],obj[1],obj[2],obj[3]])).reshape(-1,4),axis=0)
			# Tracker에서 감지한 Target의 경계박스 안에 YOLO의 사람 좌표가 있는지 확인
			if tx > obj[0] and ty > obj[1] and bx < obj[2] and by < obj[3] :
				inbox.append(True)
			else:
				inbox.append(False)
		# 좌표의 거리 중 가장 큰 값을 maxi에 저장
		maxi=neartarget.max(-1)
		#mini=neartarget.min(-1)
		if len(neartarget)==0:
			target=[obj]
		else:
			# minimum: 가장 먼 좌표의 거리가 가장 작은 값을 가지는 Index
			minimum=maxi.argmin()
			try:
				target=[person[inbox.index(True)]]
			except:
				target=[person[minimum]]
			# Target 설정
			targetOn=setTracker(img,tracker,target,targetOn)
	return targetOn,target

# Target 경계박스 표현
def setTracker(img,tracker,target,targetOn=1):
	tx,ty,bx,by=int(target[0][0]),int(target[0][1]),int(target[0][2]),int(target[0][3])
	if targetOn==0:
		bbox=(tx,ty,bx-tx,by-ty)
		ok=tracker.init(img,bbox)
		targetOn=1
	cv2.rectangle(img,(tx,ty),(bx,by),(0,255,255),3)
	cv2.putText(img,"Target",(bx, ty-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)
	return targetOn
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

# Tracker 갱신
def updateTracker(img,result,tracker,prevtarget):
	# Tracker 갱신 [ok: 추적 성공, 실패 -> True,False]
	ok,bbox=tracker.update(img)
	bbox=[int(bbox[0]),int(bbox[1]),int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])]
	check=0
	print('bbox:',bbox)
	print('prevtarget:',prevtarget)
	if ok:
		check,target=get_target(img,result,tracker,bbox)
		#cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
		#cv2.putText(img,"Tracker",(bbox[2], bbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
	else:
		# 추적에 실패 했을 경우 이전 프레임의 Target 위치를 기반으로 YOLO의 사람과의 거리를 대조해 계속 추적하도록 한다
		check,target=get_target(img,result,tracker,prevtarget[0])
	return target

def pickling(tOn,hd,tr,detect,t):
	dopick={'on':tOn, 'hand':hd, 'track':tr, 'detect_result':detect, 'target':t}
	buffer = pickle.dumps(dopick,protocol=2)
	return buffer

def unpick_img(unpick):
	i = unpick["image"]
	array = np.frombuffer(i, dtype=np.dtype('uint8'))
	return cv2.imdecode(array,1)

def net_check(client):
	data=-1
	print('net_check')
	while data==-1:
		try:
			data = client.getData()
		except Exception as ex: 
	    		print('Can Not Connect Server', ex)
	    		sleep(5)
	print('return data')
	return data

def main():
	# 네트워크 통신 속성 설정
	client=netInfo()
	client.setClient(5001)
	track,hand,rec_info,prevtarget,detect=[],[],[],[],[]
	# Tracker 선언 및 설정
	tracker=createTrackerByName("KCF")
	prevTime,targetOn,angleStack,yawTime,net_flag=0,0,0,0,-1
	# YOLO Save 파일 설정
	tf_hand=resultTF("hand")
	tf_person=resultTF("person")
	tf_detect=resultTF("detect")
	print('setting up on capThread.')
	tf_hand.start()
	tf_person.start()
	tf_detect.start()
	print('starting up on capThread.')
	while True:
		while net_flag==-1:
			net_flag = net_check(client)
		while True:
			try:
				print('send to get')
				upd = client.sendData(b'get')
				if len(upd)==14:
					continue
				# 받은 데이터 압축 해제
				unpick=pickle.loads(upd)
				img=unpick_img(unpick)
				print('success unpick')
				net_targeton,net_hand,net_track,net_detectResult,net_target=0,None,None,None,None
				if targetOn==0:
					# YOLO를 기반 손 탐지 결과
					result=tf_hand.getBuffer(img)
					gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
					# 손의 움직임을 표현할 마스킹 이미지(image 위에 바로 표현하면 인식에 방해될 수 있기때문에 따로 생성)
					mask=np.ones_like(img,np.uint8)
					# Display FPS
					prevTime=dp_fps(img,prevTime)
					# hand_rectangle: 손의 경계박스를 표현하고 좌표값을 리스트 리턴
					hand=hand_rectangle(img,result)
					# track = 동일손의 중앙 좌표값끼리 묶어놓은 리스트(3차원)
					track=distance_Box(track,center_Box(hand))
					# 마스킹 이미지에 위 손의 좌표값들을 선으로 표현
					cv2.polylines(mask,([np.int32(tr) for tr in track]),False,(255,255,0),3)
					# shape_detect(): 선들이 도형을 이루는지, 그 도형이 사각형이면 타겟 인식으로 전환  
					rec_info,targetOn,prevtarget=shape_detect(img,mask,rec_info,targetOn,tracker,tf_person)
					net_targeton=targetOn
					print('set targetOn:',net_targeton)
					if targetOn==1:
						# give targetFLAG
						net_target=prevtarget[0]
						data=pickling(net_targeton,net_hand,net_track,net_detectResult,net_target)
						client.sock.sendto(data,client.address)

					# else : give hand, track, 
					else:
						net_hand=hand
						net_track=track
						data=pickling(net_targeton,net_hand,net_track,net_detectResult,net_target)
						client.sock.sendto(data,client.address)
					img=cv2.add(img,mask)

				else:
					result=tf_person.getBuffer(img)
					detect_result=tf_detect.getBuffer(img)
					# Display FPS
					prevTime=dp_fps(img,prevTime)
					prevtarget=updateTracker(img,result,tracker,prevtarget)
					# give detect_result & prevtarget[0]
					net_targeton=targetOn
					net_detectResult=detect_result
					net_target=prevtarget[0]
					data=pickling(net_targeton,net_hand,net_track,net_detectResult,net_target)
					client.sock.sendto(data,client.address)
				cv2.imshow('server',img)
				key=cv2.waitKey(10)
				if ord('q')==key:
					client.sock.close()
					exit(0)
			except Exception as ex:
				print('Server_Error!! ', ex)
	print('== Turn over ==')

if __name__=='__main__':
	main()

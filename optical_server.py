#!/home/lee/anaconda2/bin/python
#coding=utf8
from darkflow.net.build import TFNet
import cv2
import numpy as np
import urllib
import time
from PIL import Image
from mss import mss

#Flow Option
hp="C:\\Users\\lee\\AnacondaProjects"
'''options={"model": hp+"\\darkflow-master\\cfg\\handvocV2-yolo.cfg",
 "pbLoad":hp+"\\darkflow-master\\built_graph\\handvocV2-yolo.pb",
 "metaLoad":hp+ "\\darkflow-master\\built_graph\\handvocV2-yolo.meta",
 "threshold":0.1, "gpu":0.7}'''
options={"model": hp+"\\darkflow-master\\cfg\\handvocV2-yolo.cfg",
 "load":hp+"\\darkflow-master\\ckpt\\handvocV2-yolo-51750.data-00000-of-00001",
 "threshold":0.1, "gpu":0.7,"labels":hp+"\\darkflow-master\\labels.txt"}

# Params for ShiTomasi corner Detection
feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (10,10), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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

# Find New POINT
def m_point(old_gray,person):
	mask = np.zeros_like(gray)
	mask[:]=255
	for x,y in [np.float32(tr[-1]) for tr in track]:
		cv2.circle(mask,(x,y),5,0,-1)
	p = cv2.goodFeaturesToTrack(old_gray,mask=mask,**feature_params)
	if p is not None:
		for t_x,t_y,b_x,b_y in person:
			for x,y in np.float32(p).reshape(-1,2):
				if(t_x<=x and t_y <=y and b_x >= x and b_y>= y):
					track.append([[x,y]])

def return_p(old_gray,gray):
	p0r= np.float32([tr[-1] for tr in track]).reshape(-1,1,2)
	if(len(track)==0):
		p0r=cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)
	p1,st,err = cv2.calcOpticalFlowPyrLK(old_gray,gray,p0r,None,**lk_params)
	p1r,st,err = cv2.calcOpticalFlowPyrLK(gray,old_gray,p1,None,**lk_params)
	d= abs(p1r-p0r).reshape(-1,2).max(-1)
	# 이전 포인트와의 거리가 30 이하인 점이면 True 	
	good= d<30
	return p1,d,good

def dp_fps(img,prevTime):
	# Display FPS
	curTime=time.time()
	sec=curTime - prevTime
	prevTime=curTime
	fps = 1/(sec)
	cv2.putText(img,"FPS : %0.1f"%fps,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	return prevTime

def add_record(track,p1,good):
	# track 갱신
	n_tr=[]
	for tr,(a,b),good_flag in zip(track,p1.reshape(-1,2),good):
		if not good_flag:
			continue
		# good == True면 track에 새로운 좌표 추가 :: que
		tr.append([a,b])
		# 한 영역의 좌표들의 개수가 20개 초과시 처음 좌표 삭제 :: que
		if len(tr)>20:
			del tr[0]
		n_tr.append(tr)
	return n_tr

#Capture x,y position
mon={'top':150, 'left':150, 'width':800, 'height':600}
#window ScreenCapture
sct=mss()

if __name__ == '__main__':
	tfnet=TFNet(options)
	print("============ Video Start ============")
	old_img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
	old_gray=cv2.cvtColor(old_img,cv2.COLOR_RGB2GRAY)
	prevTime=0
	track=[]
	hand=[]
	cnt=0
	while(True):
		img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
		gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		result=tfnet.return_predict(img)
		print(result)
		# Display FPS
		prevTime=dp_fps(img,prevTime)
		# Optical Flow
		p1,d,good=return_p(old_gray,gray)
		track=add_record(track,p1,good)
		print(len(track))
		hand=draw_rec(img,result)
		cv2.polylines(img,[np.int32(tr) for tr in track],False,(255,255,0),3)
		cv2.imshow('video',img)
		old_gray=gray.copy()
		cnt += 1
		# cnt Frame 마다, 새로운 Point 찾기 (안 그러면 같은 포인트로만 Draw)
		if cnt % 10 == 0:
			m_point(old_gray,hand)
			cnt=0
		if ord('q')==cv2.waitKey(10):
			exit(0)
	print('== Turn over ==')


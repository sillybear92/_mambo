from darkflow.net.build import TFNet
import cv2
import numpy as np
import time
from mss import mss
import sys
import drawMov


def dp_fps(img,prevTime):
	# Display FPS
	curTime=time.time()
	sec=curTime - prevTime
	prevTime=curTime
	fps = 1/(sec)
	cv2.putText(img,"FPS : %0.1f"%fps,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	return prevTime

def draw_rec(img,result):
	n_detect=[]
	for obj in result:
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

def get_target(img,result,tracker,obj):
	target=[]
	targetFlag=0
	inbox=[]
	neartarget=np.array([]).reshape(-1,4)
	if result==-1:
				return 
	person=draw_rec(img,result)
	if len(person)==0:
		target=[obj]
	else:
		for [tx,ty,bx,by] in person:
			neartarget=np.append(neartarget,abs(np.array([tx,ty,bx,by])-np.array([obj[0],obj[1],obj[2],obj[3]])).reshape(-1,4),axis=0)
			if tx > obj[0] and ty > obj[1] and bx < obj[2] and by < obj[3] :
				inbox.append(True)
				continue
			inbox.append(False)
		#neartarget=abs(np.array([[[tx,ty,bx,by]] for [tx,ty,bx,by] in person]).reshape(-1,4)\
		# - np.array([obj[0],obj[1],obj[2],obj[3]]).reshape(-1,4))
		maxi=neartarget.max(-1)
		#mini=neartarget.min(-1)
		if len(neartarget)==0:
			target=[obj]
		else:
			print(neartarget)
			minimum=maxi.argmin()
			try:
				target=[person[inbox.index(True)]]
			except:
				target=[person[minimum]]
			setTracker(img,tracker,target)
	return targetFlag,target

def updateTracker(tracker,img,result,prevtarget):
	ok,bbox=tracker.update(img)
	bbox=[int(bbox[0]),int(bbox[1]),int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])]
	check=0
	print('bbox:',bbox)
	print('prevtarget:',prevtarget)
	if ok:
		check,target=get_target(img,result,tracker,bbox)
		cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
		cv2.putText(img,"obj",(bbox[2], bbox[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
	else:
		check,target=get_target(img,result,tracker,prevtarget[0])
	return target

def setTracker(img,tracker,target,targetOn=1):
	tx,ty,bx,by=int(target[0][0]),int(target[0][1]),int(target[0][2]),int(target[0][3])
	#bbox=(tx,ty,bx-tx,by-ty)
	#ok=tracker.init(img,bbox)
	cv2.rectangle(img,(tx,ty),(bx,by),(0,255,255),3)
	cv2.putText(img,"Target",(bx, ty-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)


def main():
	hp="F:\\AnacondaProjects"
	optionPerson={"pbLoad":hp+"\\darkflow-master\\built_graph\\yolo180905.pb",
	 "metaLoad":hp+ "\\darkflow-master\\built_graph\\yolo180905.meta",
	 "threshold":0.4, "gpu":0.7}
	personTf=TFNet(optionPerson)

	#Capture x,y position
	mon={'top':150, 'left':150, 'width':800, 'height':600}
	#window ScreenCapture
	sct=mss()
	img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
	result=personTf.return_predict(img)
	tracker=createTrackerByName("CSRT")
	person=draw_rec(img,result)
	prevTime=0
	check=0
	prevtarget=[]
	while(len(person)==0):
		img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
		result=personTf.return_predict(img)
		person=draw_rec(img,result)
	print(person[0])
	bbox=int(person[0][0]),int(person[0][1]),int(person[0][2]-person[0][0]),int(person[0][3]-person[0][1])
	ok=tracker.init(img,bbox)
	check=setTracker(img,tracker,person)
	prevdraw=[]
	print(person[0])
	mov=drawMov.drawMov(person[0])
	while(True):
		img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
		result=personTf.return_predict(img)
		# Display FPS
		prevTime=dp_fps(img,prevTime)
		prevtarget=updateTracker(tracker,img,result,prevtarget)
		mov.drawCenter(img)
		mov.drawLine(img,prevtarget[0])
		cv2.imshow('Image', img)
		if ord('q')==cv2.waitKey(10):
			exit()



if __name__=='__main__':
	main()

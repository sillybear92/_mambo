from darkflow.net.build import TFNet
import cv2
import numpy as np
import time
from mss import mss
import sys


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
		#if(confidence>0.1):
			#cv2.rectangle(img,(top_x, top_y),(bottom_x, bottom_y), (0, 255, 0),2)
			#cv2.putText(img, label+' - ' + str(  "{0:.0f}%".format(confidence * 100) ),(bottom_x, top_y-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)
		if label == 'person':
			n_hand.append([top_x,top_y,bottom_x,bottom_y])
	return n_hand

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
	if result==-1:
				return 
	person=draw_rec(img,result)
	neartarget=abs(np.array([[[tx,ty,bx,by]] for [tx,ty,bx,by] in person]).reshape(-1,4)\
	 - np.array([obj[0],obj[1],obj[2],obj[3]]).reshape(-1,4))
	maxi=neartarget.max(-1)
	mini=neartarget.min(-1)
	if len(neartarget)==0:
		target=[obj]
	else:
		minimum=maxi.argmin()
		target=[person[minimum]]
		setTracker(img,tracker,target)
	'''if count==1:
		for [tx,ty,bx,by],goodflag in zip(person,good):
			if goodflag==False:
				continue
			print(goodflag)
			target=[[tx,ty,bx,by]]
			setTracker(img,tracker,target)
			targetFlag=1
	else:
		if len(good)==0:
			target=[obj]
		else:
			minimum=mini.argmin()
			target=[person[minimum]]
			setTracker(img,tracker,target)'''
	return targetFlag,target

def updateTracker(tracker,img,result,prevtarget):
	ok,bbox=tracker.update(img)
	bbox=[int(bbox[0]),int(bbox[1]),int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])]
	check=0
	print('bbox:',bbox)
	print('prevtarget:',prevtarget)
	if ok:
		check,target=get_target(img,result,tracker,bbox)
	else:
		check,target=get_target(img,result,tracker,prevtarget[0])
	return target

def setTracker(img,tracker,target,targetOn=1):
	tx,ty,bx,by=int(target[-1][0]),int(target[-1][1]),int(target[-1][2]),int(target[-1][3])
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
	tracker=createTrackerByName("KCF")
	person=draw_rec(img,result)
	prevTime=0
	check=0
	prevtarget=[]
	print(person[-1])
	bbox=int(person[-1][0]),int(person[-1][1]),int(person[-1][2]-person[-1][0]),int(person[-1][3]-person[-1][1])
	ok=tracker.init(img,bbox)
	check=setTracker(img,tracker,person)
	while(True):
		img=cv2.cvtColor(np.array(sct.grab(mon)),cv2.COLOR_RGBA2RGB)
		result=personTf.return_predict(img)
		# Display FPS
		prevTime=dp_fps(img,prevTime)
		prevtarget=updateTracker(tracker,img,result,prevtarget)
		cv2.imshow('Image', img)
		cv2.waitKey(1)


if __name__=='__main__':
	main()

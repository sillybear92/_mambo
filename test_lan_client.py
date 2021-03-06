#coding=utf8

import socket
import cv2
import pickle
import numpy as np
from mss import mss
from threading import Thread, Lock
import sys
from lib.netInfo2 import netInfo
#from lib.TTS import TTS
#from TTS_SECRET import TTS_SECRET
#from lib import stt
from multiprocessing import Process
import time
#from lib.test_drawMov import drawMov




class VideoCapture(Thread):
	def __init__(self):
		Thread.__init__(self)
		self.running = True
		self.lock = Lock()
		self.sct = mss()
		self.mon = {'top':100, 'left':90, 'width':800, 'height':600}
		self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
		#self.preconnect()

	def stop(self):
		self.running = False

	def getImage(self):
		self.lock.acquire()
		img = cv2.cvtColor(np.array(self.sct.grab(self.mon)),cv2.COLOR_RGBA2RGB)
		check,encode_image = cv2.imencode('.jpg', img, self.encode_param)
		self.lock.release()
		return img,encode_image

# Display FPS
def dp_fps(img,prevTime):
	curTime=time.time()
	sec=curTime - prevTime
	prevTime=curTime
	fps = 1/(sec)
	cv2.putText(img,"FPS : %0.1f"%fps,(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	return prevTime

def draw_hand(img,hand):
	for hd in hand:
		cv2.rectangle(img,(hd[0], hd[1]),(hd[2], hd[3]), (0, 255, 0),2)
		cv2.putText(img, "Hand",(hd[2], hd[1]-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)

def draw_target(img,tr):
	cv2.rectangle(img,(tr[0], tr[1]),(tr[2], tr[3]), (0, 255, 255),3)
	cv2.putText(img, "Target",(tr[2], tr[1]-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),2)

def draw_rectangle(img,result):
	for obj in result:
		label = obj['label']
		confidence = obj['confidence']
		top_x = obj['topleft']['x']
		top_y = obj['topleft']['y']
		bottom_x = obj['bottomright']['x']
		bottom_y = obj['bottomright']['y']
		cv2.rectangle(img,(top_x, top_y),(bottom_x, bottom_y), (0, 255, 0),2)
		cv2.putText(img, label+' - ' + str(  "{0:.0f}%".format(confidence * 100) ),
			(bottom_x, top_y-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)

def main():
	client=netInfo()
	client.setServer('bbik.iptime.org',1005)
#	client.setServer('192.168.0.14',5001)
	client.setClient(6666)
	cap=VideoCapture()
	print('setting up on capThread.')
	cap.start()
	print('starting up on capThread.')
	#starting up on capThread.
	print('starting up on %s port %s\n' % client.server_address)
	'''
	mov=drawMov()
	while not mov.droneCheck:
		mov.droneConnect()
		print('Power On Drone')
	'''
	# Speech recognize
	#speech=Process(target=stt.run)
	#speech.start()
	#print('Start STT PROCESS--')
	#tts=TTS()
	#tts.setID(TTS_SECRET.id,TTS_SECRET.secret)
	targetOn,hand,mask,angleStack,yawTime,prevTime,target,tcnt=0,None,None,0,0,0,None,0
	client.sock.sendto(b'connect server',client.server_address)
	print('send to connect server1')

	while(True):
		#mov.update()
		try:
			data, address = client.sock.recvfrom(65507)
			if data == None:
				print('data is None')
				continue
			img,encode_img = cap.getImage()
			if data == b'get':
				print('data is get')
				dopick = {'image' : encode_img.tobytes()}
				buffer = pickle.dumps(dopick,protocol=2)
				if buffer is None:
					print('buffer is None')
					continue
				if len(buffer) > 65507:
					print(len(buffer))
					print("The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
					client.sock.sendto(b'FAIL',address)
					continue
				print('send buffer')
				client.sock.sendto(buffer, address)
				print('success send data:', len(buffer))
				if not targetOn and hand is not None:
					draw_hand(img,hand)
					img=cv2.add(img,mask)
				elif targetOn and target is not None:
					draw_target(img,target)
					print('get-> draw Target')

			else:
				print('unpickle data')
				unpick = pickle.loads(data)
				targetOn=unpick['on']
				hand=unpick['hand']
				track=unpick['track']
				detect_result=unpick['detect_result']
				target=unpick['target']
				print('on:',targetOn,'   hand: ',hand)
				print('target: ',target)
				if targetOn==0:
					mask=np.ones_like(img,np.uint8)
					cv2.polylines(mask,([np.int32(tr) for tr in track]),False,(255,255,0),3)
					draw_hand(img,hand)
					img=cv2.add(img,mask)
				else:
					#tts.mostRisk(detect_result,[target],img,95)
					'''
					if tcnt==0 and mov.mambo.sensors.flying_state == 'landed':
						mov.droneStart()
						print(">>>>>>>>>>>>>>>>>>>>>>>>take off<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
						print(u'드론 이륙!!!!!!!!!!!!!!')
						tcnt += 1
					mov.setTarget(target)
					print('Set Target')
					mov.drawCenter(img)
					mov.drawLine(img)
					print('Draw Center,Line')
					if tcnt > 20:
						angleStack,yawTime=mov.adjPos(img,target,angleStack,yawTime)
						print(tcnt)
					else:
						tcnt+=1
					'''
					draw_rectangle(img,detect_result)
					draw_target(img,target)
					print('draw Target')
					
			prevTime=dp_fps(img,prevTime)
			cv2.imshow('client',img)
			if ord('q')==cv2.waitKey(10):
				#mov.droneStop()
				cap.join()
				client.sock.close()
				exit(0)
		except Exception as ex:
			print('Client_Error!! ', ex)
			client.sock.sendto(b'connect server',client.server_address)
			print('send to connect server2')
		#if (speech.is_alive()==False) or (mov.droneBattery < 2):
			#mov.droneStop()
			#exit(0)
			#print('DISCONNECT !!!!!!!!!!!!')

	print("Bye..")

if __name__=='__main__':
	main()



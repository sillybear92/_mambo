#coding=utf8

import socket
import cv2
import pickle
import numpy as np
from mss import mss
from threading import Thread, Lock
import sys
from lib.netInfo2 import netInfo
from lib.TTS import TTS
#from lib.drawMov import drawMov
from TTS_SECRET import TTS_SECRET
from lib import stt
from multiprocessing import Process




class VideoCapture(Thread):
	def __init__(self):
		Thread.__init__(self)
		self.running = True
		self.image = None
		self.lock = Lock()
		self.sct = mss()
		self.mon = {'top':150, 'left':150, 'width':800, 'height':600}
		self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
		#self.preconnect()

	def stop(self):
		self.running = False

	def getImage(self):
		self.lock.acquire()
		img = cv2.cvtColor(np.array(self.sct.grab(self.mon)),cv2.COLOR_RGBA2RGB)
		check,self.image = cv2.imencode('.jpg', img, self.encode_param)
		self.lock.release()
		return self.image


def main():
	#host = '0.0.0.0'
	#port = 5001
	#port = 6666
	client=netInfo()
	#client.setServer('bbik.iptime.org',1005)
	client.setServer('192.168.0.14',5001)
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
	speech=Process(target=stt.run)
	speech.start()
	print('Start STT PROCESS--')
	tts=TTS()
	tts.setID(TTS_SECRET.id,TTS_SECRET.secret)
	angleStack,yawTime=0,0
	while(True):
		try:
			data, address = client.sock.recvfrom(65507)
			img = cap.getImage()
			if(data == b'get'):
				dopick = {'image' : img.tobytes()}
				buffer = pickle.dumps(dopick,protocol=2)
				if buffer is None:
					continue
				
				if len(buffer) > 65507:
					print(len(buffer))
					print("The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
					client.sock.sendto(b'FAIL',address)
					continue
				client.sock.sendto(buffer, address)

			elif not data == None:
				unpick = pickle.loads(data)
				targetOn=unpick['targetOn']
				hand=unpick['hand']
				track=unpick['track']
				detect_result=unpick['detect_result']
				target=unpick['target']
				if not targetOn:
					mask=np.ones_like(img,np.uint8)
					cv2.polylines(mask,([np.int32(tr) for tr in track]),False,(255,255,0),3)
					for hd in hand:
						cv2.rectangle(img,(hd[0], hd[1]),(hd[2], hd[3]), (0, 255, 0),2)
						cv2.putText(img, "hand",(hd[2], hd[1]-5),  cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0, 255, 0),1)
					img=cv2.add(img,mask)
				else:
					#mov.drawCenter(img)
					#mov.drawLine(img)
					#angleStack,yawTime=mov.adjPos(img,target,angleStack,yawTime)
					detect=draw_rectangle(img,detect_result)
					#tts.mostRisk(detect_result,[target],img,mov.droneBattery)

			cv2.imshow('client',img)
			#mov.update()
			if ord('q')==key:
				#mov.droneStop()
				cap.join()
				client.sock.close()
				exit(0)
		except:
			print("time out")
	print("Bye..")

if __name__=='__main__':
	main()



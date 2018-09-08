#coding=utf8

import socket
import cv2
import pickle
import numpy as np
from mss import mss
from threading import Thread, Lock
from darkflow.net.build import TFNet
import sys



class VideoCapture(Thread):
	def __init__(self,option=None):
		Thread.__init__(self)
		self.running = True
		self.image = None
		self.lock = Lock()
		self.sct = mss()
		self.mon = {'top':150, 'left':150, 'width':800, 'height':600}
		self.result = None
		self.option = option
		self.tf=None
		self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
		self.preconnect()

	def stop(self):
		self.running = False

	def getBuffer(self):
		self.lock.acquire()
		img=cv2.cvtColor(np.array(self.sct.grab(self.mon)),cv2.COLOR_RGBA2RGB)
		check,self.image = cv2.imencode('.jpg', img, self.encode_param)
		self.result = self.tf.return_predict(img)
		self.lock.release()
		return self.image,self.result

	def getImage(self):
		self.lock.acquire()
		img = cv2.cvtColor(np.array(self.sct.grab(self.mon)),cv2.COLOR_RGBA2RGB)
		check,self.image = cv2.imencode('.jpg', img, self.encode_param)
		self.lock.release()
		return self.image

	def connectTf(self):
		self.tf=TFNet(self.getOption(self.option))
		
		'''
	def run(self):
		while(self.running):			
			self.lock.acquire()
			self.image = cv2.cvtColor(np.array(self.sct.grab(self.mon)),cv2.COLOR_RGBA2RGB)
			self.result = self.tf.return_predict(self.image)
			cv2.imshow("Stream Window",self.image)
			cv2.waitKey(1)
			self.lock.release()
			'''

	def getOption(self,option):
		hp="F:\\AnacondaProjects"
		optionHand={"pbLoad":hp+"\\darkflow-master\\built_graph\\hand180905.pb",
		 "metaLoad":hp+ "\\darkflow-master\\built_graph\\hand180905.meta",
		 "threshold":0.4, "gpu":0.7}
		optionPerson={"pbLoad":hp+"\\darkflow-master\\built_graph\\yolo180905.pb",
		 "metaLoad":hp+ "\\darkflow-master\\built_graph\\yolo180905.meta",
		 "threshold":0.4, "gpu":0.7}
		tfOptions = {"hand" : optionHand, "person" : optionPerson}
		return tfOptions[option]

	def preconnect(self):
		self.tf=TFNet(self.getOption(self.option))
		img=cv2.cvtColor(np.array(self.sct.grab(self.mon)),cv2.COLOR_RGBA2RGB)
		self.result = self.tf.return_predict(img)



def main():
	host = '0.0.0.0'
	port = 5001
	capHand = VideoCapture("hand")
	capPerson = VideoCapture("person")
	print('setting up on capThread.')
	capHand.start()
	capPerson.start()
	print('starting up on capThread.')
starting up on capThread.
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	server_address = (host, port)
	print('starting up on %s port %s\n' % server_address)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	sock.bind(server_address)
	while(True):
		data, address = sock.recvfrom(4)
		if(data == b'hdg'):
			img,result = capHand.getBuffer()
			dopick = {'image' : img.tobytes(), 'result' : result}
			buffer = pickle.dumps(dopick,protocol=2)
			if buffer is None:
				continue
			
			if len(buffer) > 65507:
				print(len(buffer))
				print("The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
				sock.sendto(b'FAIL',address)
				continue
			
			sock.sendto(buffer, address)

		elif(data == b'psg'):
			img,result = capPerson.getBuffer()
			dopick = {'image' : img.tobytes(), 'result' : result}
			buffer = pickle.dumps(dopick,protocol=2)
			if buffer is None:
				continue
			
			if len(buffer) > 65507:
				print(len(buffer))
				print("The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
				sock.sendto(b'FAIL',address)
				continue
			
			sock.sendto(buffer, address)

		elif(data == b'get'):
			img = capPerson.getImage()
			dopick = {'image' : img.tobytes()}
			buffer = pickle.dumps(dopick,protocol=2)
			if buffer is None:
				continue
			
			if len(buffer) > 65507:
				print(len(buffer))
				print("The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
				sock.sendto(b'FAIL',address)
				continue
			
			sock.sendto(buffer, address)

		elif(data == "quit"):
			capHand.stop()
			capPerson.stop()

	print("Bye..")
	#capHand.join()
	capPerson.join()
	sock.close()

if __name__=='__main__':
	main()



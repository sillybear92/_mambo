import socket
import cv2
import numpy as np
import sys
import pickle

class netInfo:
	def __init__(self):
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.host = None
		self.port = None
		self.server_address = None
		self.client_host = '0.0.0.0'
		self.client_port = 5001
		self.sock.bind((self.client_host,self.client_port))
		self.sock.settimeout(0.5)

	def setServer(self,host,port):
		self.host=host
		self.port=port
		self.server_address=(self.host,self.port)

	def sendData(self,message):
		self.sock.sendto(message,self.server_address)
		try:
			data,server = self.sock.recvfrom(65507)
		except socket.timeout as err:
			print("Timeout !! again !! ")
			return -1,-1
		print("Fragment size: {}".format(len(data)))
		if len(data) == 4:
			while(len(data)==4):
				self.sock.sendto(message,self.server_address)
				try:
					data,server = self.sock.recvfrom(65507)
				except socket.timeout as err:
					print("Timeout !! again !! ")
					return -1,-1
				print("Fragment size: {}".format(len(data)))
		unpick = pickle.loads(data)
		i = unpick["image"]
		array = np.frombuffer(i, dtype=np.dtype('uint8'))
		img = cv2.imdecode(array,1)
		try:
			result = unpick["result"]	
		except:
			result = None
		return img, result
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
		self.client_host = None
		self.client_port = None
		self.address=None

	def setServer(self,host,port):
		self.host=host
		self.port=port
		self.server_address=(self.host,self.port)

	def setClient(self,port):
		self.client_host = '0.0.0.0'
		self.client_port = port
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.sock.bind((self.client_host,self.client_port))
		self.sock.settimeout(0.5)


	def getData(self):
		try:
			data,self.address = self.sock.recvfrom(65507)
		except Exception as ex:
			print('Net_getData_Error!! ', ex)
			return -1
		print("Fragment size: {}".format(len(data)))
		return data

	def sendData(self,message):
		data=-1
		while(data==-1):
			self.sock.sendto(message,self.address)
			data=self.getData()
		return pickle.loads(data)
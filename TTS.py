# -*- coding: utf-8 -*-
#coding=utf8
## TTS만 Thread
import os
import sys
import urllib
import pyglet
import time
import cv2
from imutils import paths
import numpy as np
from scipy.spatial import distance
from multiprocessing import Process
import pygame


class TTS():
	def __init__(self):
		self.client_id = None
		self.client_secret = None
		self.speaker = "mijin" 
		self.speed = "0" 
		self.val = {
					"speaker": self.speaker,
    	    		"speed":self.speed,
		}
		self.headers = {
    	    		"X-NCP-APIGW-API-KEY-ID" : self.client_id,
    	    		"X-NCP-APIGW-API-KEY" : self.client_secret
		}
		self.p=None
		self.call_battery_10=False
		self.call_battery_5=False

	def setID(self,id,secret):
		self.client_id=id
		self.client_secret=secret
		self.headers = {
    	    		"X-NCP-APIGW-API-KEY-ID" : self.client_id,
    	    		"X-NCP-APIGW-API-KEY" : self.client_secret
		}

	def run(self,f):
		if self.p == None:	
			self.val["text"]='전방에 '+f+'가 있습니다.'
			self.play_audio()

		else:	# self.p != None
			if pygame.mixer.music.get_busy()==0:
				self.stop_audio()
				self.p.terminate()
				self.p=None

	def play_audio(self):
		data = urllib.parse.urlencode(self.val).encode("utf-8")
		url = "https://naveropenapi.apigw.ntruss.com/voice/v1/tts"
		request = urllib.request.Request(url, data, self.headers)
		response = urllib.request.urlopen(request)
		rescode = response.getcode()
		if(rescode==200):
			print("TTS save")
			response_body = response.read()
			try:
				with open('test.mp3','wb') as f:
					f.write(response_body)
			except:
				print("Already open File")
			self.p=Process(target=self.audio())
			self.p.start()

		else:
			print("Error Code:" + rescode)

	def call_battery(self,batteryGauge):
		battery=batteryGauge
		if self.p == None:
			if (battery < 10) and (self.call_battery_10==False):
				self.val["text"]='드론의 배터리가 얼마남지 않았습니다.'
				self.play_audio()
				self.call_battery_10=True
			elif (battery < 5) and (self.call_battery_5 == False):
				self.val["text"]='드론의 배터리가 거의다 소모되었습니다. 잠시후 비상착륙합니다.'
				self.play_audio()
				self.call_battery_5=True

	def stop_audio(self):
		pygame.mixer.music.stop()
		pygame.mixer.quit()
		pygame.quit()
		try:
			os.remove('test.mp3')
		except:
			print(u'제거 할수 없습니다.')

	def audio(self):
		pygame.mixer.init(16000, -16, 1, 2048)
		pygame.mixer.music.load('test.mp3')
		pygame.mixer.music.play(0)

	def recog_traffic(self,trafficlight):
		hsv=cv2.cvtColor(trafficlight,cv2.COLOR_RGB2HSV)
		lowGreen=np.array([30,100,100])
		upGreen=np.array([55,255,255])
		lowRed=np.array([115,100,100])
		upRed=np.array([125,255,255])
		maskGreen=cv2.inRange(hsv,lowGreen,upGreen)
		maskRed=cv2.inRange(hsv,lowRed,upRed)
		gr=cv2.bitwise_and(trafficlight,trafficlight,mask=maskGreen)
		rd=cv2.bitwise_and(trafficlight,trafficlight,mask=maskRed)
		cv2.imshow('original',trafficlight)
		cv2.imshow('green',gr)
		cv2.imshow('red',rd)
		if rd.sum()>gr.sum():
			print(u'#####################빨간색')
		elif rd.sum()<gr.sum():
			print(u'#####################초록색')
		else:
			print(u'감지 할 수 없음')

	#risk factor distinction
	def mostRisk(self,obstacle,prevtarget,img,battery):
		print("func in")
		risk_distance=10000
		per_cdn=distance.euclidean((prevtarget[0][0],prevtarget[0][1]),(prevtarget[0][2],prevtarget[0][3]))
		#dict in list
		print(obstacle)
		#Battery Check
		self.call_battery(battery)
		for o in obstacle:
			#50% over
			if o['confidence']>0.4:
				print("in")
				xt,yt,xb,yb=o['topleft']['x'],o['topleft']['y'], o['bottomright']['x'],o['bottomright']['y']
				print("width: {0},height: {1}".format(xb-xt,yb-yt))
				if o['label']=="traffic":
					trafficlight=img[yt:yb,xt:xb]
					self.recog_traffic(trafficlight)
				else:
					x_mid = distance.euclidean(xt,xb)
					obs_cdn=(x_mid,yb) #obstacle_coordinate #############수정
					#closet risk
					if risk_distance>distance.euclidean(obs_cdn, per_cdn):
						risk_distance=distance.euclidean(obs_cdn, per_cdn)				
						risk_factor=o['label']
						if risk_factor=='bicycle':
							risk_factor='자전거'
						elif risk_factor=='car':
							risk_factor='자동차'
						elif risk_factor=='deskchair':
							risk_factor='책상과 의자'
						elif risk_factor=='bollard':
							risk_factor='볼라드'
				
					if risk_distance<500: # HELP HELP 
						#print(risk_distance, risk_factor)
						self.run(risk_factor)

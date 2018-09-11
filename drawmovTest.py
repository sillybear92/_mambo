import cv2
import detect_client
import drawMov
import numpy as np

def main():
	client = detect_client.netInfo()
	host='mambo.iptime.org'
	port=1005
	person=[]
	prevPerson=[]
	prevtime=0
	client.setServer(host,int(port))
	while(len(prevPerson)==0):
		img,result=client.sendData(b'psg')
		if result==-1:
			continue
		person=detect_client.draw_rectangle(img,result)
		mov=drawMov.drawMov(person[0])
		if len(person)>0:
			prevPerson=person[0]
	while(True):	
		img,result=client.sendData(b'psg')
		if result==-1:
			continue
		prevtime=detect_client.dp_fps(img,prevtime)
		person=detect_client.draw_rectangle(img,result)
		if(len(prevPerson)>0 and len(person)>0):
			mov.drawCenter(img)
			mov.drawLine(img,prevPerson)
			cv2.rectangle(img,(person[0][0],person[0][1]),(person[0][2],person[0][3]),(0,255,255),2)
			prevPerson=person[0]
		cv2.imshow('video',img)

		if ord('q')==cv2.waitKey(10):
			exit(0)

if __name__=='__main__':
	main()


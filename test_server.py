#!/home/bbik/anaconda2/bin/python
# coding: utf-8


# For debugging :
# - run the server and remember the IP of the server
# And interact with it through the command line:
# echo -n "get" > /dev/udp/192.168.0.39/1080
# echo -n "quit" > /dev/udp/192.168.0.39/1080

import socket
import cv2
from threading import Thread, Lock
import sys
import numpy as np
import netifaces as ni
from PIL import Image
from mss import mss

if(len(sys.argv) != 2):
        print("Usage : {} interface".format(sys.argv[0]))
        print("e.g. {} eth0".format(sys.argv[0]))
#        sys.exit(-1)


def get_ip(interface_name):
		"""Helper to get the IP adresse of the running server"""
		import netifaces as ni
		ip = ni.ifaddresses(interface_name)[2][0]['addr']
		print(ip)
		return ip  # should print "192.168.100.37"

debug = True
jpeg_quality = 95
host='0.0.0.0'
port = 5001
sct=mss()
mon={'top':150, 'left':150, 'width':800, 'height':600}


class VideoGrabber(Thread):
        """A threaded video grabber.
        
        Attributes:
        encode_params (): 
        cap (str): 
        attr2 (:obj:`int`, optional): Description of `attr2`.
        
        """
        def __init__(self, jpeg_quality):
                """Constructor.
                Args:
                jpeg_quality (:obj:`int`): Quality of JPEG encoding, in 0, 100.
                
                """
                Thread.__init__(self)
                self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                self.running = True
                self.buffer = None
                self.lock = Lock()
		self.winName='Capure Window'
		self.winPosx=150
		self.winPosy=118

        def stop(self):
                self.running = False

        def get_buffer(self):
                """Method to access the encoded buffer.
                Returns:
                np.ndarray: the compressed image if one has been acquired. None otherwise.
                """
                if self.buffer is not None:
                        self.lock.acquire()
                        cpy = self.buffer.copy()
                        self.lock.release()
                        return cpy
                
        def run(self):
                while self.running:
                        img = np.array(sct.grab(mon))
                        cv2.imshow(self.winName,img)
			cv2.waitKey(1)
                        # JPEG compression
                        # Protected by a lock
                        # As the main thread may asks to access the buffer
                        self.lock.acquire()
                        result, self.buffer = cv2.imencode('.jpg', img, self.encode_param)
                        self.lock.release()
		cv2.destroyAllWindows()
		self.stop()


grabber = VideoGrabber(jpeg_quality)
grabber.start()

running = True

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Bind the socket to the port
server_address = (host, port)

print('starting up on %s port %s\n' % server_address)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(server_address)

while(running):
        data, address = sock.recvfrom(4)
        print(data==b'get')
        if(data == b'get'):
                buffer = grabber.get_buffer()
                if buffer is None:
                        continue
                if len(buffer) > 65507:
                        print("The message is too large to be sent within a single UDP datagram. We do not handle splitting the message in multiple datagrams")
                        sock.sendto("FAIL",address)
                        continue
                # We sent back the buffer to the client
                sock.sendto(buffer.tobytes(), address)
        elif(data == "quit"):
                grabber.stop()
        
print("Quitting..")
grabber.join()
sock.close()

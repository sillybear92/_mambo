#!/home/lee/anaconda2/bin/python
# coding: utf-8

import socket
import cv2
import numpy as np
import sys
import netifaces as ni



if(len(sys.argv) != 3):
    print("Usage : {} hostname port".format(sys.argv[0]))
    print("e.g.   {} 192.168.35.62 1080".format(sys.argv[0]))
    sys.exit(-1)


cv2.namedWindow("Image")

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
host = sys.argv[1]
port = int(sys.argv[2])
server_address = (host, port)
client_host='0.0.0.0'
#client_host=ni.ifaddresses(ni.interfaces()[2])[2][0]['addr']
client_port=5001
print(client_host)
sock.bind((client_host,client_port))
sock.settimeout(0.5)
print "server_address is ",server_address[0],server_address[1]
while(True):
    sock.sendto("get", server_address)
    try:
    	data, server = sock.recvfrom(65507)
    except socket.timeout as err:
   	print("Timeout!!! again")
    	continue
    print("Fragment size : {}".format(len(data)))
    if len(data) == 4:
        # This is a message error sent back by the server
        if(data == "FAIL"):
            continue
    array = np.frombuffer(data, dtype=np.dtype('uint8'))
    img = cv2.imdecode(array, 1)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("The client is quitting. If you wish to quite the server, simply call : \n")
print("echo -n \"quit\" > /dev/udp/{}/{}".format(host, port))

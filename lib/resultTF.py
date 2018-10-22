from threading import Thread, Lock
from darkflow.net.build import TFNet
import sys

class resultTF(Thread):
	def __init__(self,option=None,img=None):
		Thread.__init__(self)
		self.running = True
		self.image = img
		self.lock = Lock()
		self.result = None
		self.option = option
		self.tf=None
		self.preconnect(self.img)

	def stop(self):
		self.running = False

	def getBuffer(self,img):
		self.lock.acquire()
		self.result = self.tf.return_predict(img)
		self.lock.release()
		return self.result

	def connectTf(self):
		self.tf=TFNet(self.getOption(self.option))

	def getOption(self,option):
		hp="F:\\AnacondaProjects"
		optionHand={"pbLoad":hp+"\\darkflow-master\\built_graph\\hand180905.pb",
		 "metaLoad":hp+ "\\darkflow-master\\built_graph\\hand180905.meta",
		 "threshold":0.4, "gpu":0.7}
		optionPerson={"pbLoad":hp+"\\darkflow-master\\built_graph\\yolo180905.pb",
		 "metaLoad":hp+ "\\darkflow-master\\built_graph\\yolo180905.meta",
		 "threshold":0.4, "gpu":0.7}
		optionDetect={"pbLoad":hp+"\\darkflow-master\\built_graph\\detect180909.pb",
		 "metaLoad":hp+ "\\darkflow-master\\built_graph\\detect180909.meta",
		 "threshold":0.4, "gpu":0.7}
		tfOptions = {"hand" : optionHand, "person" : optionPerson, "detect": optionDetect}
		return tfOptions[option]

	def preconnect(self,img):
		self.tf=TFNet(self.getOption(self.option))
		self.result = self.tf.return_predict(img)

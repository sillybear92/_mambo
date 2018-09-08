from darkflow.net.build import TFNet
import cv2
import numpy as np

hp="F:\\AnacondaProjects"
optionHand={"model":hp+"\\darkflow-master\\cfg\\hand180905.pb",
 "metaLoad":hp+ "\\darkflow-master\\built_graph\\hand180905.meta",
 "threshold":0.4, "gpu":0.7, "train", "trainer":"adam", ""}
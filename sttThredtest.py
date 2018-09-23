import stt
from multiprocessing import Process
from multiprocessing import Pool

proc=[]
p=Pool(processes=1)
speechList=['mambo']
stopFlag=p.map(stt.run,iterable=speechList)
while True:
	print(stopFlag)

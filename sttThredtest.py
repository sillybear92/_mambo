import stt
from multiprocessing import Process
def main():
	p=Process(target=stt.run)
	p.start()
	while True:
		if (p.is_alive()):
			continue
		print(u'STT와 연결 끊김')

if __name__=='__main__':
	main()


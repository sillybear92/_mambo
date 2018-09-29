# 위험해! 맘보 
## 소개
  드론 패럿 맘보 기본 패키지에 카메라를 부착하여 Yolo 기반으로 객체 인식 처리하여 전방에 위험물체를 알려준다. <br>
  본 프로그램에서 사용되는 맘보는 "Follow Me" 기능이 없어 카메라의 영상을 기반으로 "Follow me" 하도록 프로그래밍 하였다.<br>
  그리고 쫓아 다닐 타겟은 보행자를 기준으로 하였으며, 타겟 설정은 이륙하기 전에 실시한다.<br>
  타겟 설정 기준은 "손으로 네모를 그리는 사람"이며, 3~4회 지속적으로 네모를 그리면 타겟으로 인식한다.<br>
  또, 타겟을 쫓아다니며 전방에 위험물체를 인식하며 타겟과의 거리가 기준보다 가까워지면 음성으로 알려준다.<br>
  착륙 명령은 STT를 사용한 음성처리를 사용했으며 "맘보 그만해"를 마이크로 말하면 된다.<br>
  
## 사용방법
  해당 프로그램은 공간이 넓은 곳에서 실행하십시오.
  1. 서버 프로그램을 실행 시킨 후 클라이언트를 실행시킨다. <br>
  2. 패럿 맘보와 연결이 되면 카메라 영상 스트리밍이 실시된다. <br>
  3. 카메라 앞에서 네모를 3~4회 그리면 타겟으로 인식되고, 인식됨과 동시에 맘보가 이륙한다. <br>
  4. 맘보가 나를 추적하기 시작하며, 전방의 위험물체가 나와의 거리가 기준보다 가까워 지면 음성으로 알려준다. <br>
  5. 본 프로그램에서는 손, 사람, 자동차, 자전거, 볼라드, 신호등, 책상, 의자가 인식된다. <br>
  6. 착륙 음성 명령은 "맘보 그만해" 이며, 드론의 배터리가 10% 미만일 때 비상착륙 안내메세지가 나온다. <br>
  7. 비상 종료키는 'q' & 'ctrl + c' 이다. <br>
  
## 사용환경
   * ### 아나콘다
      - [Install](https://www.anaconda.com/download/)
      - Anaconda 3.6 설치 후 3.5 로 다운그레이드 하십시오.
      - conda install python=3.5
   * ### CUDA
      - [Install](https://developer.nvidia.com/cuda-90-download-archive)
      - NVIDIA 그래픽 카드 및 GPU를 사용한다면 설치해주십시오.
      - CUDA 9.0 버전으로 설치하십시오.
   * ### Cudnn
      - [Install](https://developer.nvidia.com/cudnn)
      - CUDA 버전에 맞는 Cudnn 라이브러리를 설치하십시오.
   * ### Tensorflow
      - [Tensorflow Documentation](https://www.tensorflow.org/install/?hl=ko)
      - GPU를 사용한다면 Tensorflow-GPU를 설치하십시오.
   * ### OpenCV
      - conda install opencv -c conda-forge 
   * ### Pyparrot
      - [Pyparrot Documentation](https://pyparrot.readthedocs.io)
      - 패럿 맘보를 제어하는 Python 코드 입니다.
   * ### Naver TTS
      - 네이버에서 제공하는 음성처리 라이브러리.
      - [TTS Documentation]
      - private key를 detect_client.py의 246번재 라인에 추가하십시오.
   * ### Google STT
      - 구글에서 제공하는 음성인식 라이브러리.
      - [STT Documentation](https://cloud.google.com/speech-to-text/docs/quickstart)
      - private key Json 파일을 다운로드 하십시오.
      - 위 Json 파일의 경로를 lib/stt.py 의 183번째 라인에 추가하십시오.
   * ### Yolo pb 파일
      - [pb & meta Files Download](https://drive.google.com/open?id=1-wnS9bJGFwhNmluJnpWyGnCoC59xqD3M)
      - Yolo 체크포인트 파일.
      - 위 pb 파일을 다운로드 받아 파일의 경로를 detect_server.py 의 60번째 라인에 각각 추가하십시오.
      
## 라이센스
   * [GPLv3 Liscense](https://github.com/sillybear92/testmambo/blob/master/LICENSE)
   

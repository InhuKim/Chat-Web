# Chat-Web


# 공부 주제

0. 프로젝트를 구현을 위해서 공부중인 주제
   - subject classification : AI hub의 일상대화 데이터셋을 통해서 입력 문장이 어떤 주제인지 판별.
   - graph_Data_study : graph 데이터 포맷에 대한 공부. pytorch geometric을 활용하여 데이터셋을 만들고 데이터를 확인하는 방법 공부. dataset을 파일로 만들어 디스크 공간을 활용하는 방식이어서 colab에서는 활용성이 떨어져 메모리 방식으로 변경 필요.


# 프로젝트 목적

0. 목적
   - 해당 프로젝트는 개인 프로젝트로 연구하고 싶은 연구주제와 관련된 이론을 공부하고 실증하기 위한 기능 구현과 최종적으로 만들고 싶은 각각의 개개인 1명에게 완벽히 개인화시킬 수 있는 AI 어시스트를 구현하기 위한 공부 및 연습 프로젝트

<br>

1. 챗봇의 구현
- Graph Neural networks를 활용한 기능 구현
  - 구글의 검색엔진과 데이터베이스 knowledge graph 기반으로 구축되어 있음.
  - 질의응답, 추천시스템 기능과 graphDB를 연계하여 기능을 구현.
  - 여기서 기능의 구현을 knowledge graph에 연계하려는 이유는 단순히 학습을 통한 확률 모델인 implicit AI가 아니라 explicit AI로 구현해 보기 위함.
  - knowledge graph를 explicit AI로 구현하기 위한 방법으로 선택한 이유는 대학원에서 온톨로지, knowledge graph가 이런 방식이라고 배웠기 때문.

<br>

 2. 웹 적용
- 딥러닝 모델을 적용해서 서비스의 형태로 보여 줄 웹페이지(Django)

<br>

 3. TTS, STT
- siri, 빅스비와 같이 인공지능 비서로서의 역할을 구현하기 위한 음성인식, 출력 기능 


<br>

# 프로젝트 개요

### 1. 대화 주제 분류

 - AI Hub 데이터셋을 활용하여 분류 모델 제작.
 - 학습은 colab에서 실행하여 완성된 모델을 가져옴.
 - 데이터통신 포맷은 json으로 고정.
 - 대화 주제 분류는 화자의 의도를 파악하기 위한 용도.
 - 이후에 이 주제에 맞는 대답을 할 수 있도록 자연어 생성 준비.
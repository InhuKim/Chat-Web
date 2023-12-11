# Chat-Web


# 공부 주제

0. 프로젝트를 구현을 위해서 공부중인 주제 (practice 폴더)
   - subject classification : AI hub의 일상대화 데이터셋을 통해서 입력 문장이 어떤 주제인지 판별.
   - graph_Data_study : graph 데이터 포맷에 대한 공부. pytorch geometric을 활용하여 데이터셋을 만들고 데이터를 확인하는 방법 공부.
   - db_processing : AIhub의 한글 데이터셋을 활용하여 vectorDB인 ChromaDB에 실제로 데이터를 임베딩하여 넣는 과정.
   - agent_example : RAG 기반 챗봇을 만들기 위해서 시범적으로 langchain을 활용하여 기능 구현.


# 프로젝트 개요

0. 목적
   - 해당 프로젝트는 개인 프로젝트로 연구하고 싶은 연구주제와 관련된 이론을 공부하고 실증하기 위한 기능 구현과 최종적으로 만들고 싶은 각각의 개개인 1명에게 완벽히 개인화시킬 수 있는 AI 어시스트를 구현하기 위한 공부 및 연습 프로젝트

<br>

1. 챗봇의 구현
- RAG(Retrieval Augmented Generation)
  - LLM과 연동하여 DB 데이터를 프롬프트에 추가하여 환각(Hallucination) 현상을 개선한 방법론
  - LLM은 chatgpt 3.5 turbo api를 활용
  - langchain을 활용하여 llm과 프롬프트를 구현
  - vectorDB는 ChromaDB 활용

<br>

 2. prompt test [TBD]
- prompt 자동 평가 툴
   - prompt enginnering을 위해서 여러 프롬프트를 작성하고 평가할 때 수동으로는 불편함.
   - 여러 종류의 프롬프트 기법을 활용하려는 모델, 데이터, 서비스 목표에 따라 사용할 자동화 된 평가 툴
   - langchain에 대한 이해도와 prompt engineering에 대한 이해도가 높아지면 구현 예정

<br>

 3. TTS, STT
- siri, 빅스비와 같이 인공지능 비서로서의 역할을 구현하기 위한 음성인식, 출력 기능
   - whisper 모델을 활용하여 STT 기능 구현 예정


<br>




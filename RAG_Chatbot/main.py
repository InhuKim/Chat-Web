import Agent.agent as ag
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
model_name='gpt-3.5-turbo'

db = ag.RAG_DB(persist_path="/home/RAG_Chatbot/DB/chroma_storage", collection_name="documents_collection")
prompte_rag = ag.RAG_PROMPT("/home/RAG_Chatbot/Agent/prompt_ex.txt").return_template()
llm = ag.Chat_LLM(model_name=model_name, api_key=os.getenv('OPENAI_API_KEY'), prompt=prompte_rag).return_model()

st.header("RAG based Chatbot(ChatGPT 3.5 turbo API)")

st.markdown("""---""")

# question_input = st.text_input("질문을 입력하세요")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = llm.run({"question" : prompt, "context" : db.semantic_query(prompt),"language" : 'korean'})
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
        
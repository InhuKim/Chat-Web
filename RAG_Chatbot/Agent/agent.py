import chromadb
from chromadb.utils import embedding_functions

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

class RAG_DB():
    def __init__(self, persist_path, collection_name):
        clinet = chromadb.PersistentClient(path=persist_path)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-mpnet-base-v2")
        self.collection = clinet.get_or_create_collection(name=collection_name, embedding_function=sentence_transformer_ef)

    def semantic_query(self, query):
        result = self.collection.query(
            query_texts=query,
            n_results=3
        )

        return " ".join(result['documents'][0]).strip()


template_RAG = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""

class RAG_PROMPT():
    def __init__(self, template_path):
        self.template_path = template_path

    def return_template(self):
        with open(self.template_path, 'r') as f:
            template = f.read()
        return PromptTemplate.from_template(template)
    

class Chat_LLM():

    def __init__(self, api_key, model_name, prompt):
        self.api_key =api_key
        self.model_name = model_name
        self.prompt = prompt
    
    def return_model(self):
        return LLMChain(llm=ChatOpenAI(model_name=self.model_name, temperature= 0, openai_api_key=self.api_key), prompt=self.prompt)



from typing import Any, List, Mapping, Optional, Dict
from pydantic import Field

import chromadb
from chromadb.utils import embedding_functions

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel, RobertaConfig, AutoModelForCausalLM


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

class RoBERT_Tokenizer():
    def __init__(self, config) -> None:
        super(RoBERT_Tokenizer, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.config = config

    def encode(self, text):
        inputs = self.tokenizer.encode_plus(text, return_tensors='pt', padding='max_length', max_length=self.config['max_seq_len'], truncation=True, add_special_tokens=True)
        input_ids = inputs['input_ids'][0].type(torch.long)
        attention_masks = inputs['attention_mask'][0].type(torch.long)

        return input_ids.unsqueeze(0), attention_masks.unsqueeze(0)

class RoBERTa(nn.Module):
    def __init__(self, config) -> None:
        super(RoBERTa, self).__init__()

        self.model_config = RobertaConfig.from_pretrained(config['model_name'])
        self.bert = RobertaModel(self.model_config)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=768, out_features=384),
            nn.LeakyReLU(),
            nn.Linear(in_features=384, out_features=config['out_features']),
        )
        # self.parsing_dict = label

    def forward(self, ids, masks):
        _, x = self.bert(input_ids= ids, attention_mask=masks, return_dict=False)
        logit = self.classifier(x)
        # output = logit.argmax(1).detach().cpu().numpy().tolist()

        # return [self.parsing_dict[i] for i in output][0]
        return logit

class NLU_CustomLLM(LLM):

    CFG : Dict = Field(None, alias="model_config")
    label : Dict = Field(None, alias="label dictionary")
    model : Any = None
    tokenizer : Any = None
    # out_features : int = Field(None, alias="classification count")


    def __init__(self, CFG: Dict, label: Dict):
        super(NLU_CustomLLM, self).__init__()

        self.CFG : Dict = CFG
        self.label : Dict = label

        self.model : Any = RoBERTa(self.CFG)
        # self.model = model.load_state_dict(torch.load(CFG['model_path'], map_location=device))
        self.tokenizer : Any = RoBERT_Tokenizer(self.CFG)


        self.load_model()


    def load_model(self) -> None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.load_state_dict(torch.load(self.CFG['model_path'], map_location=device))

    @property
    def _llm_type(self) -> str:
        return "custom_nlu"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # tokenizer = RoBERT_Tokenizer(CFG)
        # model = RoBERTa(CFG, label)

        # .pth 파일 경로 체크
        
        inputs = self.tokenizer.encode(prompt)
        logit = self.model(inputs[0], inputs[1])
        tag = logit.argmax(1).detach().cpu().numpy().tolist()

        return [self.label[str(i)] for i in tag][0]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"NLU model": RoBERTa}
    

class NLG_CustomLLM(LLM):

    model_name : str = Field(None, alias="model_name")
    model : Any = None
    tokenizer : Any = None
    device : Any = None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    def __init__(self, model_name : str, config : BitsAndBytesConfig = bnb_config):
        super(NLG_CustomLLM, self).__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model : Any = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=config, device_map={"":0})
        self.tokenizer : Any = AutoTokenizer.from_pretrained(model_name)

    @property
    def _llm_type(self) -> str:
        return "custom_NLG"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        inputs = self.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        input_ids = inputs['input_ids'].to(self.device)

        outputs = self.model.generate(
            input_ids = input_ids,
            max_new_tokens=256,
            generation_config=GenerationConfig(temperature=0.1, top_p=3, top_k=50, num_beams=3, repetition_penalty=1.0)
            )

        return self.tokenizer.decode(outputs[0])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"NLG model": self.model}
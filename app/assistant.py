from langchain_core.runnables import Runnable, RunnableConfig
from typing import Any, Dict, Optional
from langchain_core.runnables.utils import Input, Output
from generation.prompt_templates import get_template
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda, RunnableMap
import os
import yaml
from langchain.memory import ConversationBufferWindowMemory
from pathlib import Path
import re

# Get the root path
root_path = os.path.dirname(os.path.abspath(__file__))



# Load a .yaml or .yml file
with open(Path(root_path).parent/"generation/prompt_templates.yaml", "r",encoding="utf-8") as file:
    prompt_config = yaml.safe_load(file)


translate_prompt_Temp = get_template('translator')['translator']
history_aware_prompt = prompt_config['context_aware_prompt']['prompt']
details_agent_prompt = prompt_config['details_agent_prompt']['prompt']
translate_to_hindi = get_template('translator_english_hindi')['translator_english_hindi']

# Chat manager class to manage chat
class ChatManager:
    """ """
    def __init__(self, llm, vector_db):

        self.llm = llm
        self.vector_db = vector_db
        self.rag_chain = self._create_rag_chain()



     # Create a rag chain
    def _create_rag_chain(self):
        return create_retrieval_chain(
            self.history_aware_retriever_chain(),
            self.q_a_chain()
        )




    # history aware/context aware chain for retrieving the context from the vector database
    def history_aware_retriever_chain(self):

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("assistant", history_aware_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )



        history_aware_retriever_chain = create_history_aware_retriever(
            self.llm, self.vector_db, contextualize_q_prompt
        )

        return history_aware_retriever_chain



    # question answer chain for the answer  the  questions from the retrieved context/docs
    def q_a_chain(self):

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("assistant", details_agent_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ])

        details_agent_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        return details_agent_chain


   # combine all steps in run function that will takes query and return the reponse

    async def run(self,query,chat_history)->Any:

        translated = (
                translate_prompt_Temp | self.llm
        ).invoke({'query': query})


        translated_text = translated.content.strip()


        result = self.rag_chain.invoke({
            "input": translated_text,
            "chat_history": chat_history
        })



        translated = (
                translate_to_hindi | self.llm
        ).invoke({'query': result['answer']})
        print("----translated---",translated.content.strip())
        return  translated.content.strip()




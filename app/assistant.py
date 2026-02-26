from langchain_core.runnables import Runnable, RunnableConfig
from typing import Any, Dict, Optional
from langchain_core.runnables.utils import Input, Output
from .generation.prompt_templates import get_template
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.messages import HumanMessage, AIMessage
import os
from .settings import PROMPT_TEMPLATES_PATH 
import yaml
from langchain.memory import ConversationBufferWindowMemory
from pathlib import Path
import re



# Load a .yaml or .yml file
with open(PROMPT_TEMPLATES_PATH , "r",encoding="utf-8") as file:
    prompt_config = yaml.safe_load(file)




translate_prompt_Temp = get_template('translator')['translator']
history_aware_prompt = prompt_config['context_aware_prompt']['prompt']
details_agent_prompt = prompt_config['details_agent_prompt']['prompt']
translate_to_hindi = get_template('translator_english_hindi')['translator_english_hindi']

# Chat manager class to manage chat
class ChatManager:
    """ 
    this chat manager class will handle vector database and llm with langchain this class create the translation chain,history_aware_retriever_chain and  q_a_chain
    with langchain.
    """
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

        # ✅ Gemma doesn't support "system" role
        # Pattern: user → assistant → user → assistant...
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", history_aware_prompt),       # ✅ "user" instead of "assistant/system"
                ("assistant", "Understood."),          # ✅ dummy assistant to maintain alternating pattern
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )

        history_aware_retriever_chain = create_history_aware_retriever(
            self.llm, self.vector_db, contextualize_q_prompt
        )

        return history_aware_retriever_chain


    # question answer chain
    def q_a_chain(self):

        # ✅ Same fix: user → assistant → chat_history → user
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", details_agent_prompt),        # ✅ "user" instead of "assistant/system"
                ("assistant", "Understood."),           # ✅ dummy assistant to alternate roles
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ])

        details_agent_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        return details_agent_chain


    def _format_chat_history(self, chat_history):
        """
        Convert list of dicts to LangChain message objects.
        Ensures strict user/assistant alternation.
        """
        if not chat_history:
            return []

        formatted = []
        for msg in chat_history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'user':
                formatted.append(HumanMessage(content=content))
            elif role == 'assistant':
                formatted.append(AIMessage(content=content))

        # Ensure alternating pattern - remove consecutive same roles
        fixed = []
        for msg in formatted:
            if fixed and type(msg) == type(fixed[-1]):
                fixed[-1].content += " " + msg.content  # merge same-role messages
            else:
                fixed.append(msg)

        return fixed


 # combine all steps in run function
    async def run(self, query, chat_history) -> Any:

        # ✅ Format chat history to proper LangChain message objects
        formatted_history = self._format_chat_history(chat_history)


        # call the rag chain with formatted history
        result = self.rag_chain.invoke({
            "input": query,
            "chat_history": formatted_history   # ✅ properly formatted
        })

      
        return result['answer'].strip()

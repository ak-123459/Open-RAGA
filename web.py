import streamlit as st
import requests
import json
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

# get url endpoint
URL = os.getenv('URL')


    
  
  


# Function to make post request
def get_reponse(prompt:str,messages:list,url:str):


    payload = {'query': prompt.strip(), 'last_3_turn': messages}

    try:

        response = requests.post(url= url, data=json.dumps(payload))

        if(response.status_code==200):

           return response.json()['response']


    except requests.exceptions.RequestException as e:
        # This will catch all types of request-related exceptions

        st.error(f"an error occurred {e}")


def ChatUI():
  
    with st.container(border=True):
    
        col1, col2, col3, col4 = st.columns([2, 7, 2, 2])
    
        with col1:
    
            st.empty()
    
            st.image("assets/icon/icons8-chatbot-120.png", width=40)
            st.image("assets/icon/google-gemini-icon.png", width=40)
    
        with col2:
    
    
            st.markdown(
                """
                <h1 style='color: #1f77b4; font-size: 40px;'>
                   EVA  <span style='font-size: 20px;'>Digital ASSISTANT</span>
                </h1>
                """,
                unsafe_allow_html=True
            )
    
        with col4:
            st.markdown("")
            st.image("assets/icon/icons8-google-assistant-120.png", width=80)
    
    
    # Initialize chat history
    if "messages" not in st.session_state:
    
        st.session_state.messages = []
    
    
    # Initialize history variable
    if 'history' not in st.session_state:
        st.session_state.history = [{"role": "user", "content": ''},{"role": "assistant", "content": ''}]
    
    
    
    
    
    
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    
    
    # Accept user input
    if prompt := st.chat_input("Write message..?"):
    
    
    
    
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # st.session_state.messages.append({'role':'assistant', "content": '💡 Thinking...'})
    
        # Display user message in chat message container
        with st.chat_message("user",avatar=":material/person:"):
            st.markdown(prompt)
    
        # Empty placeholder
        placeholder = st.empty()
    
        # Display user message in chat message container
        with placeholder.chat_message("assistant",avatar=":material/lightbulb:"):
    
            st.markdown('Thinking...')
    
    
    
        # Get the response
        response = get_reponse(prompt.strip(), st.session_state.history[-6:], URL+"/chat")
    
    
        if (response):
    
            placeholder.empty()
    
            with st.chat_message("assistant",avatar="✨"):
    
                st.markdown(response)
    
    
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
        placeholder.empty()



if not URL:
  
  print("URL is empty in environment variable")
  
else:

  respone = requests.get(url= URL.strip())

  if(respone.status_code==200):
    
        ChatUI()
  else:

    st.error(respone.status_code," not found...")
        


  
  



    

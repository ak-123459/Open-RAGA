from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from scripts.prepare_data import get_file_contents
import yaml
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scripts.prepare_data import get_file_contents
import torch



"""This embedder.py file contain the functions that will use to load the embedding model ,create the new embeddings and append the embeddings on 
existing vector database."""


# get the current directory root path
root_path = os.path.dirname(os.path.abspath(__file__))

# Load a .yaml or .yml file
with open(Path(root_path).parent/"config/dev/model_config.yaml", "r",encoding="utf-8") as file:
    model_config = yaml.safe_load(file)




# add model related config parameters
model_name = model_config['embedding_model']['name']  
model_path = model_config['embedding_model']['path'] 
model_keywords = model_config['embedding_model']['model_kwargs']
encode_kwargs = model_config['embedding_model']['encode_kwargs']
model_safe_tensors_path = model_config['embedding_model']['safe_tensor_path']


# Load a .yaml or .yml file
with open(Path(root_path).parent/"config/dev/app_config.yaml", "r") as file:
    app_config = yaml.safe_load(file)




docs_path  = app_config['docs_path']['chat_history']
chunk_size = app_config['retriever']['chunk_size']
chunk_overlap = app_config['retriever']['chunk_overlap']
vectorstore_path = app_config['vector_store']['persist_path']

# docs path 
chat_path  = app_config['docs_path']['chat_history']







# load huggingface embeddings model from disk

def load_embeddings_model():
    
      print("Loading embedding model..")


      embedding_model = HuggingFaceEmbeddings(
            model_name= model_path,
            model_kwargs=model_keywords,
            encode_kwargs=encode_kwargs   # important for cosine sim
        )

      print(f"Embedding Model loaded Successfully at {model_path}..")

      return embedding_model


    

"""This get_embedding_model load the embedding model if exist on local files else it will download  from source and return the embedding model."""

# Get embeddings model

def get_embedding_model(model_path=model_path):

  # Check if specify path of embedding model exist if true then directly load the embedding model.

  if(os.path.exists(model_path)):


      return  load_embeddings_model()

  # Download the model from Huggingface if it's not available offline.
  else:

      os.makedirs(model_path,exist_ok=True)  # make directory for embedding model
       
      print("Downloading embedding model..")
      
      embedding_model = SentenceTransformer(model_name)       # get model from sources

      embedding_model.save(model_path)   # save the embedd. model 
      
      print(f"Embedding Model downloaded Successfully at {model_path}..")
      
      return   load_embeddings_model()  # return model for inferences

     
     


"""This append_embeddings function will simply update the existing vector database by simply converting the documents into vectors and adding on vector database."""

# ------------ updates the existing vector database ---------------------------------------

def append_embeddings(docs_path= chat_path,vectorstore_path=vectorstore_path):

            # Split into chunks
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size
                                                  , chunk_overlap=chunk_overlap)
            docs = text_splitter.create_documents(get_file_contents(docs_path))
            
            if(os.path.exists(docs_path)):
                
                    vectorstore  = FAISS.load_local(vectorstore_path, embeddings=get_embedding_model(),dangerous_deserialization=True)
                
                    # 3. Add new texts
                    vectorstore.add_texts(docs)
                    
                    # 4. Save updated index
                    vectorstore.save_local(vectorstore_path) 
    
                    print(f"Vector updated successfully in {vectorstore_path}:- ") 
            else:

                print(f"------>>>> No {docs_path} - path exits")
                


"""This create_embedings function will simply create the embeddings and save the vector embeddings on specify path."""

# --------------- create embeddings from the new docs -------------------------

def create_embeddings(docs_path= chat_path,vectorstore_path=vectorstore_path):

       if(os.path.exists(docs_path)):

            print("Creating embeddings ...\n")
            # Split into chunks
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size
                                                  , chunk_overlap=chunk_overlap)
        
            docs = text_splitter.create_documents(get_file_contents(docs_path))  
           
            try:
                vectorstore  = FAISS.from_documents(docs, get_embedding_model())
                 
                vectorstore.save_local(vectorstore_path)
    
                print(f"Vector created successfully in {vectorstore_path}:- ")

                return vectorstore

            except Exception as e:
                
                     print(f"Something went wrong: {e}")
                
                     return None

           
           
       else:       
                   
                   print(f"documents does not exist in directory :- {docs_path}")
           
                   return None


           
       
                            
                        
        


from langchain_community.vectorstores import FAISS
import os
from retrieval.embedder import create_embeddings,get_embedding_model
from pathlib import Path
import yaml
root_path = os.path.dirname(os.path.abspath(__file__))


# set Disable SymLinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_RESUME_DOWNLOAD"] = "1"


# Load a .yaml or .yml file
with open(Path(root_path).parent/"config/dev/app_config.yaml", "r",encoding="utf-8") as file:
    app_config = yaml.safe_load(file)


vector_db = app_config['vector_store']['persist_path']

docs_path  = app_config['docs_path']['preprocessed']






# get vector store
def get_vector_store(vector_db_path=vector_db):
    
  if(os.path.exists(vector_db_path)):

        if os.listdir(vector_db_path):
            
            vector_store =  FAISS.load_local(vector_db_path,get_embedding_model() ,allow_dangerous_deserialization=True)
            
            return vector_store

        else:

            print(f"No Index file exist in path {vector_db_path}")

    

  else:
        
        os.makedirs(vector_db_path,exist_ok=True) # make database directory

        create_embeddings(docs_path,vector_db_path)

        vector_store =  FAISS.load_local(vector_db_path,get_embedding_model() ,allow_dangerous_deserialization=True)
            
        return vector_store
        
         



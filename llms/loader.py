from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,AutoModelForCausalLM
import yaml
from pathlib import Path
import os
from transformers.utils import logging




""""
this loader.py file contain the function that should be used for development not for production. below functions will download the model from third-party source
like huggingface.
"""


# Get the  parent directory
root_path = os.path.dirname(os.path.abspath(__file__))



# Load a .yaml or .yml file
with open(Path(root_path).parent/"config/dev/model_config.yaml", "r",encoding="utf-8") as file:
    model_config = yaml.safe_load(file)

 

# ---- function to get class vise llms and tokenizer
def class_vise_models(model_config,local_files_only,class_type):


    
  if(class_type=="AutoModelForCausalLM"):

            # create model
            tokenizer = AutoTokenizer.from_pretrained(model_config['name'],local_files_only=local_files_only)
      
            model = AutoModelForCausalLM.from_pretrained(model_config['name'],local_files_only=local_files_only)
      
            
            # Create Hugging Face pipeline
            pipe = pipeline(
                model_config['type'],
                model=model,
                tokenizer=tokenizer,
                max_length=model_config['parameters']['max_length'],
                do_sample=model_config['parameters']['do_sample'],
                temperature=model_config['parameters']['temperature']  )
        
            # Wrap the pipeline with LangChain
            llm = HuggingFacePipeline(pipeline=pipe)
      
            return llm

  if(class_type=="AutoModelForSeq2SeqLM"):


             # create model
            tokenizer = AutoTokenizer.from_pretrained(model_config['name'],local_files_only=local_files_only)
      
            model = AutoModelForSeq2SeqLM.from_pretrained(model_config['name'],local_files_only=local_files_only)
      
            
            # Create Hugging Face pipeline
            pipe = pipeline(
                model_config['type'],
                model=model,
                tokenizer=tokenizer,
                max_length=model_config['parameters']['max_length'],
                do_sample=model_config['parameters']['do_sample'],
                temperature=model_config['parameters']['temperature']  )
        
            # Wrap the pipeline with LangChain
            llm = HuggingFacePipeline(pipeline=pipe)
      
            return llm

  else:

      print(f"No model Loaded for class type :- {class_type}")


    

# get llm from huggingface
def get_llm_dev(local_files_only=False,class_type='AutoModelForCausalLM'):
    
    model_class = ['AutoModelForSeq2SeqLM',"AutoModelForCausalLM"]
    
    if(class_type not in model_class):

      print("Model class is not valid try classes :- 'AutoModelForSeq2SeqLM' , 'AutoModelForCausalLM' ")
        
    else:
        
       for model_infos in model_config['inference_models']:
     
           if(model_infos['class']==class_type):
    
              return  class_vise_models(model_infos,local_files_only,class_type)
               
        
         
        











# LLM for productions 

def get_llm_prod():
       # create model
    tokenizer = AutoTokenizer.from_pretrained(model_config['inference_model']['name'])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_config['inference_model']['name'])
    
    
    
    # Create Hugging Face pipeline
    pipe = pipeline(
        model_config['inference_model']['type'],
        model=model,
        tokenizer=tokenizer,
        max_length=model_config['inference_model']['max_length'],
        do_sample=model_config['inference_model']['do_sample'],
        temperature=model_config['inference_model']['temperature']
    )

    # Wrap the pipeline with LangChain
    return   HuggingFacePipeline(pipeline=pipe)




import yaml
import os
from pathlib import Path
import yaml
from langchain_core.language_models import BaseChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA



# Root directory files path
root_path = os.path.dirname(os.path.abspath(__file__))

# Load a .yaml or .yml file
model_config_path = str(Path(root_path).parent)+"/config/prod/model_config.yaml"



"""below is the model manager class that will get llm instance that deployed on  runpod,Nvidia,Aws or self host-server.   """


# model manager to init the inference model

class ModelManager:
    def __init__(self, config_path=model_config_path):
        self.models = {}
        self.configs = self._load_config(config_path)

    def _load_config(self, path):
        with open(path, 'r',encoding="utf-8") as f:
            return yaml.safe_load(f)['inference_models']

    def get_model(self,provider, name,key):
        if name in self.models:
            return self.models[name]


        for cfg in self.configs:

            if cfg['name'] == name and provider == 'nvidia':


                try:

                    llm = ChatNVIDIA(
                        model=cfg['name'],
                        api_key=key,
                        temperature=cfg['temperature'],
                        max_tokens=cfg['max_tokens']
                    )

                    self.models[name] = llm
                    print(f"[INFO] Model '{name}' from provider 'nvidia' loaded successfully.")


                    return llm

                except Exception as e:

                    print(f"[ERROR] Failed to load model '{name}' from provider 'nvidia': {e}")

                    return None

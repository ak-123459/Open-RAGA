from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_CONFIG_PATH = BASE_DIR /"config" /  "prod" / "model_config.yaml"
DOCS_PATH = BASE_DIR / "data" / "preprocessed" /
EMBD_MODEL_DIR = BASE_DIR / "src" / "embedder" / "model_checkpoints" /
VECTOR_STORE_PATH = BASE_DIR / "data" / "vector_db" / "knowledge_base" /


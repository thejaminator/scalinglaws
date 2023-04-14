import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
# data/lm_agree_statements.jsonl
lm_agree_statements_jsonl_path = Path("data/lm_agree_statements.jsonl")

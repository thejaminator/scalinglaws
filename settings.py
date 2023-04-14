import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file
OPENAI_KEY = os.getenv("OPENAI_KEY", "")

"""Generation results"""

# data/lm_agree_statements.jsonl
lm_agree_statements_jsonl_path = Path("data/lm_agree_statements.jsonl")
# data/lm_disagree_statements.jsonl
lm_disagree_statements_jsonl_path = Path("data/lm_disagree_statements.jsonl")

"""Preference results"""
preference_agree_statements_jsonl_path = Path("data/preference_agree_statements.jsonl")
preference_agree_statements_csv_path = Path("data/preference_agree_statements.csv")
preference_disagree_statements_jsonl_path = Path(
    "data/preference_disagree_statements.jsonl"
)
preference_disagree_statements_csv_path = Path(
    "data/preference_disagree_statements.csv"
)

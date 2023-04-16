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

"""Preference results zero shot"""
preference_agree_zeroshot_jsonl_path = Path("data/preference_agree_zeroshot.jsonl")
preference_agree_zeroshot_csv_path = Path("data/preference_agree_zeroshot.csv")
preference_disagree_zeroshot_jsonl_path = Path(
    "data/preference_disagree_zeroshot.jsonl"
)
preference_disagree_zeroshot_csv_path = Path("data/preference_disagree_zeroshot.csv")

"""Preference results cot"""
preference_agree_cot_jsonl_path = Path("data/preference_agree_cot.jsonl")
preference_agree_cot_csv_path = Path("data/preference_agree_cot.csv")
preference_disagree_cot_jsonl_path = Path("data/preference_disagree_cot.jsonl")
preference_disagree_cot_csv_path = Path("data/preference_disagree_cot.csv")

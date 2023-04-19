from scalinglaws.agree_statements_generation import run_generate_agree
from scalinglaws.agree_statements_generation_cot import stage_one_generate_agree_cot
from scalinglaws.disagree_statements_generation import run_generate_disagree
from scalinglaws.disagree_statements_generation_cot import stage_one_generate_disagree_cot
from scalinglaws.format_for_graphs import (
    format_for_final_inference,
    stage_three_format_and_filter,
)
from scalinglaws.preference_cot import stage_two_preferences_cot
from scalinglaws.preference_zero_shot import run_preferences_zero_shot

if __name__ == "__main__":
    agree_generations_n = 4000
    disagree_generations_n = 1200
    stage_one_generate_agree_cot(n_completions=agree_generations_n)
    stage_one_generate_disagree_cot(n_completions=disagree_generations_n)
    stage_two_preferences_cot(cot_n=8, limit=agree_generations_n)
    stage_three_format_and_filter()

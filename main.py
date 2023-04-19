from scalinglaws.agree_statements_generation import run_generate_agree
from scalinglaws.agree_statements_generation_cot import run_generate_agree_cot
from scalinglaws.disagree_statements_generation import run_generate_disagree
from scalinglaws.disagree_statements_generation_cot import run_generate_disagree_cot
from scalinglaws.format_for_graphs import format_for_final_inference, format_for_all_formatters
from scalinglaws.preference_cot import run_preferences_cot
from scalinglaws.preference_zero_shot import run_preferences_zero_shot

if __name__ == "__main__":
    # run_generate_agree(n_completions=100)
    # run_generate_agree_cot(n_completions=4000)
    # run_generate_disagree_cot(n_completions=1200)
    run_preferences_cot(cot_n=8, limit=99999999)
    format_for_all_formatters()

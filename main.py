from scalinglaws.agree_statements_generation import run_generate_agree
from scalinglaws.disagree_statements_generation import run_generate_disagree
from scalinglaws.format_for_graphs import format_main
from scalinglaws.preference_zero_shot import run_preferences_zero_shot

if __name__ == "__main__":
    run_generate_agree(n_completions=6000)
    run_generate_disagree(600)
    run_preferences_zero_shot()
    format_main(zero_shot_final_input=False)

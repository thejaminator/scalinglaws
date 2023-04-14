from scalinglaws.agree_statements_generation import run_generate_agree
from scalinglaws.disagree_statements_generation import run_generate_disagree
from scalinglaws.format_for_graphs import format_main
from scalinglaws.preference import run_agree_and_disagree_preferences

if __name__ == "__main__":
    run_generate_agree(n_completions=6000)
    run_generate_disagree(600)
    run_agree_and_disagree_preferences()
    format_main()

from scalinglaws.agree_statements_generation import main_generate_agree
from scalinglaws.format_for_graphs import format_main
from scalinglaws.preference_truth import main_preference

if __name__ == "__main__":
    main_generate_agree(n_completions=200)
    main_preference()
    format_main()

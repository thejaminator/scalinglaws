from scalinglaws.agree_statements_generation_cot import stage_one_generate_agree_cot
from scalinglaws.disagree_statements_generation_cot import stage_one_generate_disagree_cot
from scalinglaws.format_for_graphs import stage_three_format_and_filter
from scalinglaws.plot_accuracies import step_three_evaluate_and_create_all_plots
from scalinglaws.plot_sycophancy import plot_all_sycophancy
from scalinglaws.preference_cot import stage_two_preferences_cot

if __name__ == "__main__":
    agree_generations_n = 4000
    disagree_generations_n = 1200
    stage_one_generate_agree_cot(n_completions=agree_generations_n)
    stage_one_generate_disagree_cot(n_completions=disagree_generations_n)
    stage_two_preferences_cot(cot_n=8, limit=agree_generations_n)
    stage_three_format_and_filter()
    step_three_evaluate_and_create_all_plots()
    plot_all_sycophancy()

vanilla_models = ["ada", "babbage", "curie", "davinci"]
feedme_models = [
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-001",
]
other_rlhf = ["text-davinci-002", "text-davinci-003"]
all_davinci = ["davinci", "text-davinci-001", "text-davinci-002", "text-davinci-003"]

def truncate_model_name(model_name: str) -> str:
    # take the truncate name e.g. text-ada-001 -> ada
    truncated_model_name = model_name.split("-")[1]
    return truncated_model_name
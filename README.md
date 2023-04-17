

## To use eval pipeline
python -m eval_pipeline.main --dataset-path "/data/statements_filtered.csv" --exp-dir /evalresults --models "text-ada-001 text-babbage-001 text-curie-001 text-davinci-001" --task-type classification_acc --batch-size 100
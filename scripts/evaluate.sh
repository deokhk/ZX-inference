python -u inference.py \
    --batch_size 16 \
    --device "0" \
    --seed 42 \
    --model_name_or_path "./models/text2sql_mt0_ckpt" \
    --model_class "bigscience/mt0-base" \
    --dev_filepath "./data/sample_spider_seq2seq.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --output "predicted_sql.txt"
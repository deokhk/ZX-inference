import os
import json
import random 
import torch
import argparse
import logging 
import torch.optim as optim
import torch.nn as nn

from tqdm.auto import tqdm
from tokenizers import AddedToken

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers.trainer_utils import set_seed
from utils.load_dataset import Text2SQLDataset

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


logger = logging.getLogger(__name__)


def main(opt):
    set_seed(opt.seed)

    logger.info(f"Model class: {opt.model_class}")
    logger.info(f"Batch size: {opt.batch_size}")

    import time
    start_time = time.time()
        
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, AutoTokenizer):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    dev_dataset = Text2SQLDataset(
        dir_ = opt.dev_filepath,
        mode = "eval"
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    
    # Extract gold sqls, which will be used after 
    with open(opt.dev_filepath, "r", encoding = 'utf-8') as f:
        dev_data = json.load(f)
    
    gold_questions = []
    gold_sqls = []

    for data in dev_data:
        gold_questions.append(data["input_sequence"].split("utterance:")[1].split(" | ")[0].strip())
        gold_sqls.append(data["output_sequence"].split("<sql>")[1].strip())


    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    # initialize model

    logger.info("Loading model...")
    model = model_class.from_pretrained(opt.model_name_or_path)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_sqls = []
    for batch in tqdm(dev_dataloder, desc="Inferencing.."):
        batch_inputs = [data[0] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 256,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = opt.num_beams,
                num_return_sequences = opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])

            batch_size = model_outputs.shape[0]
            for batch_id in range(batch_size):
                
                pred_sequence = tokenizer.decode(model_outputs[batch_id, 0, :], skip_special_tokens = True)

                pred_sql = pred_sequence.split("<sql>")[-1].strip()
                pred_sql = pred_sql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")
                    
                predict_sqls.append(pred_sql)
    
    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)
    
    # save results
    with open(opt.output, "w", encoding = 'utf-8') as f:
        for pred in predict_sqls:
            f.write(pred + "\n")
    
    end_time = time.time()
    logger.info("Text-to-SQL inference spends {}s.".format(end_time-start_time))

    logger.info("Now showing some examples of predicted SQLs, along with gold SQLs:")
    
    count = 0

    for id, (question, pred_sql, gold_sql) in enumerate(zip(gold_questions, predict_sqls, gold_sqls)):

        logger.info("=============================================================")
        logger.info(f"Question: {question}")
        logger.info(f"Predicted SQL: {pred_sql}")
        logger.info(f"Gold SQL: {gold_sql}")

        count +=1 
        if count == 5:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--device', default="0")
    parser.add_argument('--batch_size', type = int, default = 8)
    parser.add_argument("--model_class", type = str, default = "t5-small")
    parser.add_argument("--model_name_or_path", type = str, default = "./checkpoints/t5-small")

    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    
    parser.add_argument('--dev_filepath', type = str, default = "data/preprocessed_data/resdsql_dev.json",
                        help = 'file path of test2sql dev set.')

    parser.add_argument("--output", type = str, default = "predicted_sql.txt")
    opt = parser.parse_args()
    main(opt)
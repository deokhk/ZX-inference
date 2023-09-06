import argparse
import os
import json
import torch 
import shutil 
import wandb 
import logging 

from text2sql import _test

logger = logging.getLogger(__name__)



def parse_option():
    parser = argparse.ArgumentParser("command line arguments for selecting the best ckpt.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size. Note that this is a effective batch size')
    parser.add_argument('--device', type = str, default = "2",
                        help = 'the id of used GPU device.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "./models/text2sql",
                        help = 'save path of fine-tuned text2sql models.')
    parser.add_argument('--model_name_or_path', type=str, default= "mt5",
                        help="Type of model used for evaluation")
    parser.add_argument('--eval_results_path', type = str, default = "./eval_results/text2sql",
                        help = 'the evaluation results of fine-tuned text2sql models.')
    parser.add_argument('--mode', type = str, default = "eval",
                        help='eval.')
    parser.add_argument('--dev_filepath', type = str, default = "./data/pre-processing/resdsql_test.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type = str, default = "./data/spider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "./data/spider/database",
                        help = 'file path of database.')
        
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--output", type = str, default = "predicted_sql.txt")
    
    parser.add_argument("--cross_eval_every_epoch", action="store_true", help="Enable for cross evaluation every epoch")

    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name for wandb logging")
    parser.add_argument("--wandb_log", action="store_true", help="Enable for logging to wandb")

    opt = parser.parse_args()

    return opt

    
if __name__ == "__main__":
    opt = parse_option()
    
    if opt.wandb_log:
        wandb_exp_name = opt.exp_name if opt.exp_name else "eval"
        wandb.init(
            project="ZX_seq2seq",
            name=wandb_exp_name,
        )

    ckpt_names = os.listdir(opt.save_path)
    ckpt_names = sorted(ckpt_names, key = lambda x:eval(x.split("-")[1]))
    
    print("ckpt_names:", ckpt_names)

    save_path = opt.save_path
    os.makedirs(opt.eval_results_path, exist_ok = True)

    spider_dev_filepath = opt.dev_filepath
    spider_original_dev_filepath = opt.original_dev_filepath

    eval_results = []
    cspider_eval_results = []
    for ckpt_name in ckpt_names:
        # if the current ckpt is being evaluated or has already been evaluated
        if "{}.txt".format(ckpt_name) in os.listdir(opt.eval_results_path):
            # is being evaluated
            with open(opt.eval_results_path+"/{}.txt".format(ckpt_name), "r") as f:
                if len(f.readlines()) == 1:
                    continue
            # has already been evaluated
            with open(opt.eval_results_path+"/{}.txt".format(ckpt_name), "r") as f:
                eval_result = json.load(f)
                eval_results.append(eval_result)
        # otherwise, we start evaluating the current ckpt
        else:
            logger.info("Start evaluating ckpt: {}".format(ckpt_name))
            with open(opt.eval_results_path+"/{}.txt".format(ckpt_name), "w") as f:
                f.write("Evaluating...")
            
            opt.save_path = save_path + "/{}".format(ckpt_name)
            opt.dev_filepath = spider_dev_filepath
            opt.original_dev_filepath = spider_original_dev_filepath

            em, exec = _test(opt)


            eval_result = dict()
            eval_result["ckpt"] = opt.save_path
            eval_result["EM"] = em
            eval_result["EXEC"] = exec

            with open(opt.eval_results_path+"/{}.txt".format(ckpt_name), "w") as f:
                f.write(json.dumps(eval_result, indent = 2, ensure_ascii = False))
            
            logger.info("ckpt name: {}".format(ckpt_name))

            eval_results.append(eval_result)
            logger.info(eval_results)

            if opt.wandb_log and not opt.cross_eval_every_epoch:
                step = int(eval(ckpt_name.split("-")[1]))
                wandb.log({"EM": eval_result["EM"], "EXEC":eval_result["EXEC"], "STEP": step}, step)

            if opt.cross_eval_every_epoch:
                logger.info("Start testing ckpt on Cspider")
                with open(opt.eval_results_path+"/{}_cspider.txt".format(ckpt_name), "w") as cf:
                    cf.write("Evaluating...")
                    
                    # Override the dev_filepath and original_dev_filepath
                    opt.dev_filepath = opt.cspider_dev_filepath
                    opt.original_dev_filepath = opt.cspider_original_dev_filepath
                    cspider_em, cspider_exec = _test(opt)


                    cspider_eval_result = dict()
                    cspider_eval_result["ckpt"] = opt.save_path
                    cspider_eval_result["EM"] = cspider_em
                    cspider_eval_result["EXEC"] = cspider_exec

                    if opt.wandb_log:
                        step = int(eval(ckpt_name.split("-")[1]))
                    
                    logger.info(cspider_eval_result)

                    cf.write(json.dumps(cspider_eval_result, indent = 2, ensure_ascii = False))
                wandb.log({"EM": eval_result["EM"], "EXEC":eval_result["EXEC"], "CSPIDER_EM": cspider_eval_result["EM"], "CSPIDER_EXEC":cspider_eval_result["EXEC"]}, step)
                cspider_eval_results.append(cspider_eval_result)

    for eval_result in eval_results:
        print("ckpt name:", eval_result["ckpt"])
        print("EM:", eval_result["EM"])
        print("EXEC:", eval_result["EXEC"])
        print("-----------")
    
    if opt.cross_eval_every_epoch:
        for cspider_eval_result in cspider_eval_results:
            print("ckpt name:", cspider_eval_result["ckpt"])
            print("EM:", cspider_eval_result["EM"])
            print("EXEC:", cspider_eval_result["EXEC"])
            print("-----------")
    

    em_list = [er["EM"] for er in eval_results]
    exec_list = [er["EXEC"] for er in eval_results]
    em_and_exec_list = [em + exec for em, exec in zip(em_list, exec_list)]

    # find best EM ckpt
    best_em, exec_in_best_em = 0.00, 0.00
    best_em_idx = 0

    # find best EXEC ckpt
    best_exec, em_in_best_exec = 0.00, 0.00
    best_exec_idx = 0

    # find best EM + EXEC ckpt
    best_em_plus_exec = 0.00
    best_em_plus_exec_idx = 0

    for idx, (em, exec) in enumerate(zip(em_list, exec_list)):
        if em > best_em or (em == best_em and exec > exec_in_best_em):
            best_em = em
            exec_in_best_em = exec
            best_em_idx = idx
        
        if exec > best_exec or (exec == best_exec and em > em_in_best_exec):
            best_exec = exec
            em_in_best_exec = em
            best_exec_idx = idx
        
        if em+exec > best_em_plus_exec:
            best_em_plus_exec = em+exec
            best_em_plus_exec_idx = idx
    
    print("Best EM ckpt:", eval_results[best_em_idx])
    print("Best EXEC ckpt:", eval_results[best_exec_idx])
    print("Best EM+EXEC ckpt:", eval_results[best_em_plus_exec_idx])

    # We only keep the checkpoint with best EM+EXEC ckpt. Delete the others.


    for idx, eval_result in enumerate(eval_results):
        if idx != best_em_plus_exec_idx:
            shutil.rmtree(eval_result["ckpt"], ignore_errors = True)

    best_ckpt_name = ckpt_names[best_em_plus_exec_idx]
    print(f"Deleted all ckpts except the best EM+EXEC ckpt : {best_ckpt_name}.")

    print(f"Now testing on Cspider and Vspider dev sets using the best EM+EXEC ckpt : {best_ckpt_name}.")

    """
    Testing on Cspider and Vspider dev sets using the best EM+EXEC ckpt
    """
    print("Start testing ckpt on Cspider: {}".format(best_ckpt_name))
    with open(opt.eval_results_path+"/{}_cspider.txt".format(best_ckpt_name), "w") as f:
        f.write("Evaluating on Cspider...")
    
    opt.save_path = save_path + "/{}".format(best_ckpt_name)
    # Override the dev_filepath and original_dev_filepath
    opt.dev_filepath = opt.cspider_dev_filepath
    opt.original_dev_filepath = opt.cspider_original_dev_filepath
    em, exec = _test(opt)


    eval_result = dict()
    eval_result["ckpt"] = opt.save_path
    eval_result["EM"] = em
    eval_result["EXEC"] = exec

    with open(opt.eval_results_path+"/{}_cspider.txt".format(best_ckpt_name), "w") as f:
        f.write(json.dumps(eval_result, indent = 2, ensure_ascii = False))
    
    print("Done.")
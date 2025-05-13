import json
import os
import argparse
from tqdm import tqdm
import base64
import torch

from mmagent.utils.chat_api import *
from mmagent.prompts import prompt_generate_captions_with_ids, prompt_generate_thinkings_with_ids
from evaluation.memory_evaluation import eval_vdcscore, eval_autodq, eval_equivalence
from mmagent.utils.general import validate_and_fix_python_list

import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gemini-1.5-pro-002")
parser.add_argument("--val_path", type=str, default="data/sft/memgen/0429/conversations/val.jsonl")
parser.add_argument("--output_dir", type=str, default="data/sft/memgen/0429/val_gen")
parser.add_argument("--val_num", type=int, default=5)
parser.add_argument("--generate", action="store_true")
args = parser.parse_args()

def generate_data(model, data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    samples = []

    with open(data_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
            
    inputs = []

    for i, sample in enumerate(tqdm(samples)):
        messages = sample[0]
        content = messages["content"]
        input = []
        sem = False
        for c in content[1:]:
            if c["type"] == "text":
                if c["text"] == "Video descriptions:":
                    sem = True
                input.append({
                    "type": "text",
                    "content": c["text"]
                })
            elif c["type"] == "image":
                input.append({
                    "type": "images/jpeg", 
                    "content": [base64.b64encode(open(c["image"], "rb").read()).decode("utf-8")]
                })
            elif c["type"] == "video":
                input.append({
                    "type": "video_base64/mp4",
                    "content": base64.b64encode(open(c["video"], "rb").read()).decode("utf-8")
                })
        
        if sem:
            input.append({
                "type": "text",
                "content": prompt_generate_thinkings_with_ids
            })
        else:
            input.append({
                "type": "text",
                "content": prompt_generate_captions_with_ids
            })
        
        inputs.append(input)
        
    messages = [generate_messages(input) for input in inputs]
    responses = parallel_get_response(model, messages, timeout=30)[0]
    
    for i, sample, response in zip(range(len(samples)), samples, responses):
        messages = sample[:1]
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        
        with open(os.path.join(output_dir, f"{i}.json"), "w") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)

def evaluate_sft(gt_path, output_dir, save_path_autodq, save_path_vdcscore, val_num=10):
    os.makedirs(os.path.dirname(save_path_autodq), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_vdcscore), exist_ok=True)
    gt_samples = []
    pred_samples = []
    idx = 0
    with open(gt_path, "r") as f:
        for line in f:
            gt_res = json.loads(line)[1]["content"][0]["text"]
            if "<speaker_" in gt_res:
                idx += 1
                continue
            gt_sample = validate_and_fix_python_list(gt_res)
            
            pred_res = json.load(open(os.path.join(output_dir, f"{idx}.json")))[1]["content"][0]["text"]
            pred_sample = validate_and_fix_python_list(pred_res)

            if gt_sample and pred_sample:
                gt_samples.append(gt_sample)
                pred_samples.append(pred_sample)
            else:
                with open(os.path.join(os.path.dirname(save_path_autodq), "error_outputs.log"), "a") as f:
                    f.write(f"Error output: {idx}, gt: {gt_res}, pred: {pred_res}\n")

            idx += 1
    
    assert len(gt_samples) == len(pred_samples)

    gt_samples = gt_samples[:val_num]
    pred_samples = pred_samples[:val_num]
    
    print("Evaluating AutoDQ...")
    precision, recall, f1 = eval_autodq(gt_samples, pred_samples, save_path_autodq)
    print("AutoDQ Evaluation:")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
    
    print("Evaluating VDCScore...")
    precision, avg_score = eval_vdcscore(gt_samples, pred_samples, save_path_vdcscore)
    print(f"VDCScore Evaluation:")
    print(f"Precision: {precision}, Avg Score: {avg_score}")
    
    print("Evaluating Equivalence...")
    precision, recall, f1 = eval_equivalence(gt_samples, pred_samples)
    print(f"Equivalence Evaluation:")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
            
            
if __name__ == "__main__":
    val_path = args.val_path
    output_dir = args.output_dir
    model = args.model
    eval_dir = os.path.join(output_dir, model)
    response_dir = os.path.join(eval_dir, "val_gen")
    val_num = args.val_num
    
    if args.generate:
        generate_data(model, val_path, response_dir)
    else:
        evaluate_sft(val_path, response_dir, os.path.join(eval_dir, f"autodq_evaluation_val_{val_num}.json"), os.path.join(eval_dir, f"vdcscore_evaluation_val_{val_num}.json"), val_num)
    

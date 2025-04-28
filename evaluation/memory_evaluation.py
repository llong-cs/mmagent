import json
from mmagent.utils.chat_api import *
from mmagent.prompts import prompt_autodq, prompt_vdcscore_generate_qas, prompt_vdcscore_answer, prompt_vdcscore_verify
from mmagent.utils.general import *

processing_config = json.load(open("configs/processing_config.json"))
MAX_RETRIES = processing_config["max_retries"]

def generate_qas_for_vdcscore(video_description):
    input = [
        {
            "type": "text",
            "content": prompt_vdcscore_generate_qas.format(video_description=video_description)
        }
    ]
    messages = generate_messages(input)
    model = "gpt-4o-2024-11-20"
    
    qas = None
    
    for _ in range(MAX_RETRIES):
        response = get_response_with_retry(model, messages)[0]
        qas = validate_and_fix_python_list(response)
        if qas is not None:
            return qas
        
    raise Exception("Failed to generate qas")

def descriptions_comparison_for_autodq(video_description_1, video_description_2):
    input = [
        {
            "type": "text",
            "content": prompt_autodq.format(video_description=" ".join(video_description_2), events=video_description_1)
        }
    ]
    messages = generate_messages(input)
    model = "gpt-4o-2024-11-20"
    
    for _ in range(MAX_RETRIES):
        response = get_response_with_retry(model, messages)[0]
        result = validate_and_fix_python_list(response)
        if result is not None:
            return result
        
    raise Exception("Failed to generate autodq result")


def eval_vdcscore(gt_desciptions, generated_descriptions, qas=None):
    if qas is None:
        qas = generate_qas_for_vdcscore(gt_desciptions)
    else:
        with open(qas, "r") as f:
            qas = json.load(f)
    
    for gt_description, generated_description in zip(gt_desciptions, generated_descriptions):
        for qa in qas:
            if qa["event"] == generated_description:
                print(qa)

def eval_autodq(gt_desciptions, generated_descriptions):
    total_precision = 0
    total_recall = 0
    precision = 0
    recall = 0
    
    for gt_description, generated_description in zip(gt_desciptions, generated_descriptions):
        precison_bases = descriptions_comparison_for_autodq(generated_description, gt_description)
        correct = 0
        for base in precison_bases:
            if base["relationship"] == "entailment":
                precision += 1
        total_precision += len(precison_bases)
        
        recall_bases = descriptions_comparison_for_autodq(gt_description, generated_description)
        correct = 0
        for base in recall_bases:
            if base["relationship"] == "entailment":
                recall += 1
        total_recall += len(recall_bases)
        
    precision = precision / total_precision
    recall = recall / total_recall
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


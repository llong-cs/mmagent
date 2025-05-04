import json
import os

from mmagent.utils.chat_api import *
from mmagent.prompts import prompt_autodq, prompt_vdcscore_generate_qas, prompt_vdcscore_answer, prompt_vdcscore_verify
from mmagent.utils.general import *

logger = logging.getLogger(__name__)

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
        qa_list = validate_and_fix_python_list(response)
        if qa_list is not None:
            qas = [
                {
                    "question": qa[0],
                    "answer": qa[1]
                } for qa in qa_list
            ]
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
        response, token = get_response_with_retry(model, messages)
        result = validate_and_fix_python_list(response)
        if result is not None:
            return result
        logger.error(f"Failed at generating {token} tokens. Retrying...")
        
    raise Exception("Failed to generate autodq result")


def eval_vdcscore(gt_desciptions, generated_descriptions, qa_file=None):
    if not os.path.exists(qa_file):
        qas = []
        for gt_description in gt_desciptions:
            qas.append(generate_qas_for_vdcscore(gt_description))
        with open(qa_file, "w") as f:
            json.dump(qas, f)
    else:
        with open(qa_file, "r") as f:
            qas = json.load(f)

    if "predicted_answer" not in qas[0][0].keys():
        for qa_list, generated_description in zip(qas, generated_descriptions):
            inputs = [
                [
                    {
                        "type": "text",
                        "content": prompt_vdcscore_answer.format(video_description=generated_description, question=qa["question"])
                    }
                ]
                for qa in qa_list
            ]
            messages = [generate_messages(input) for input in inputs]
            model = "gpt-4o-2024-11-20"
            answers = parallel_get_response(model, messages)[0]
            
            for qa, answer in zip(qa_list, answers):
                qa["predicted_answer"] = answer
        
        with open(qa_file, "w") as f:
            json.dump(qas, f)
    
    if "evaluation" not in qas[0][0].keys():
        for qa_list in qas:
            inputs = [
                [
                    {
                        "type": "text",
                        "content": prompt_vdcscore_verify.format(question=qa["question"], correct_answer=qa["answer"], predicted_answer=qa["predicted_answer"])
                    }
                ]
                for qa in qa_list
            ]
            messages = [generate_messages(input) for input in inputs]
            model = "gpt-4o-2024-11-20"
            for _ in range(MAX_RETRIES):
                evaluations = parallel_get_response(model, messages)[0]
                evaluations = [validate_and_fix_json(evaluation.replace("'", "\"")) for evaluation in evaluations]
                
                if None not in evaluations:
                    break

                raise Exception("Failed to evaluate vdcscore")
            
            for qa, evaluation in zip(qa_list, evaluations):
                qa["evaluation"] = evaluation
        
        with open(qa_file, "w") as f:
            json.dump(qas, f)
    
    correct = 0
    score = 0
    total = 0
    
    for qa_list in qas:
        for qa in qa_list:
            if qa["evaluation"]["pred"].lower().startswith("yes"):
                correct += 1
            score += float(qa["evaluation"]["score"])
            total += 1
    
    precision = correct / total
    avg_score = score / total

    return precision, avg_score

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


if __name__ == "__main__":
    gt_desciptions = [
        ["<character_0> describes the Cactus Store, located on the Lower East Side of New York City.", "The Cactus Store is situated between an old paint shop and a modern hotel.", "A man in a blue shirt and khaki pants sets up a yellow A-frame sign that reads 'Cactus Store and Other Plants'.", "Another man in a gray shirt and white shorts carries a large cactus in an orange bucket.", "The man in the gray shirt places the cactus near the entrance of the Cactus Store.", "A man walks through a narrow alley filled with plants and bamboo stalks.", "Inside a greenhouse, a man in a blue shirt and khaki pants tends to various cacti and succulents.", "He uses a small brush and a scoop to care for the plants.", "The greenhouse is filled with rows of potted cacti and succulents of different sizes and shapes, arranged on metal shelves and wooden tables.", "<character_0> welcomes viewers to the Cactus Store.", "Inside the store, <character_0> gestures towards a large plant.", "<character_0> is knowledgeable about the Cactus Store's location and offerings.", "<character_0> is welcoming and enthusiastic about the Cactus Store.", "A man in a blue shirt sets up the store for the day and tends to the plants.", "A man in a gray shirt helps set up the store by bringing in a large cactus.", "The Cactus Store is located in a slim alley on the Lower East Side of New York City.", "The Cactus Store specializes in cacti and succulent collectibles.", "The video showcases the unique and hidden nature of the Cactus Store.", "The video aims to introduce viewers to the Cactus Store and its offerings."]
    ]
    generated_descriptions = [
        ["<character_0> describes the Cactus Store, located on the Lower East Side of New York City.", "The Cactus Store is situated between an old paint shop and a modern hotel.", "A man in a blue shirt and khaki pants sets up a yellow A-frame sign that reads 'Cactus Store and Other Plants'.", "Another man in a gray shirt and white shorts carries a large cactus in an orange bucket.", "The man in the gray shirt places the cactus near the entrance of the Cactus Store.", "A man walks through a narrow alley filled with plants and bamboo stalks.", "Inside a greenhouse, a man in a blue shirt and khaki pants tends to various cacti and succulents.", "He uses a small brush and a scoop to care for the plants.", "The greenhouse is filled with rows of potted cacti and succulents of different sizes and shapes, arranged on metal shelves and wooden tables.", "<character_0> welcomes viewers to the Cactus Store.", "Inside the store, <character_0> gestures towards a large plant.", "<character_0> is knowledgeable about the Cactus Store's location and offerings.", "<character_0> is welcoming and enthusiastic about the Cactus Store.", "A man in a blue shirt sets up the store for the day and tends to the plants.", "A man in a gray shirt helps set up the store by bringing in a large cactus."]
    ]
    print("Evaluating vdcscore...")
    precision, avg_score = eval_vdcscore(gt_desciptions, generated_descriptions, "exp/0428/vdcscore_qa.json")
    print(f"Precision: {precision}, Avg Score: {avg_score}")

    print("Evaluating autodq...")
    precision, recall, f1 = eval_autodq(gt_desciptions, generated_descriptions)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")